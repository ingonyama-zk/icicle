#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h> // Only valid for lynux

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/config_extension.h"
#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"
#include "icicle/msm.h"
#include "icicle/decision_tree.h"
#include "taskflow/taskflow.hpp"
#include "icicle/backend/msm_config.h"
#include "icicle/utils/platform.h"

#include "decision_tree_params/msm_nof_cores_tree_params.h"
#include "decision_tree_params/msm_c_tree_amd_params.h"
#include "decision_tree_params/msm_c_tree_intel_params.h"
#include "decision_tree_params/msm_c_tree_arm_params.h"
#ifdef G2_ENABLED
  #include "decision_tree_params/msm_c_tree_intel_params_g2.h"
  #include "decision_tree_params/msm_c_tree_amd_params_g2.h"
  #include "decision_tree_params/msm_c_tree_arm_params_g2.h"
#endif

#ifdef MEASURE_MSM_TIMES
  #include "icicle/utils/timer.hpp"
#endif

using namespace icicle;
using namespace curve_config;

#define NOF_FEATURES_C_TREE      3
#define FIXED_SCALAR_SIZE_C_TREE 254

template <typename A, typename P>
class Msm
{
public:
  // Constructor
  Msm(const int msm_size, const MSMConfig& config) : m_msm_size(msm_size), m_config(config)
  {
    m_cpu_vendor = get_cpu_vendor();
    // TBD: for small size MSM - prefer double and add
    calc_optimal_parameters();
    // Resize the thread buckets according to optimal parameters
    m_workers_buckets.resize(m_nof_workers);
    m_workers_buckets_busy.resize(m_nof_workers);
    for (int worker_i = 0; worker_i < m_nof_workers; worker_i++) {
      m_workers_buckets[worker_i].resize(m_nof_total_buckets);
      m_workers_buckets_busy[worker_i].resize(m_nof_total_buckets);
    }
  }

  // run MSM based on pippenger algorithm
  void run_msm(const scalar_t* scalars, const A* bases, P* results, bool last_task_in_batch)
  {
    phase1_populate_buckets(scalars, bases);

    wait_last_task_completed();

    phase2_collapse_segments();

    if (last_task_in_batch) {
      phase3_final_accumulator(results);
      m_phase3_thread = nullptr;
    } else {
      m_phase3_thread = std::make_unique<std::thread>(std::thread(&Msm::phase3_final_accumulator, this, results));
    }
  }

  // Calculate the optimal number of workers based on the problem size, config and machine parameters.
  static unsigned get_optimal_nof_workers(
    const MSMConfig& config, const int msm_size, const uint32_t scalar_size, const uint32_t precompute_factor)
  {
    uint32_t nof_cores =
      config.ext && config.ext->has(CpuBackendConfig::CPU_NOF_THREADS)
        ? config.ext->get<int>(CpuBackendConfig::CPU_NOF_THREADS) // number of threads provided by config
        : std::thread::hardware_concurrency();                    // check machine properties
    if (nof_cores <= 0) {
      ICICLE_LOG_WARNING << "Unable to detect number of hardware supported threads - fixing it to 1\n";
      nof_cores = 1;
    }

    // Use decision tree to predict the optimal number of workers
    double field_size_to_fixed_size_ratio = scalar_size / FIXED_SCALAR_SIZE_CORES_TREE;
    double pcm = (double)precompute_factor;
    double msm_log_size = (double)std::log2(msm_size * field_size_to_fixed_size_ratio);
    double features[NOF_FEATURES_CORES_TREE] = {msm_log_size, pcm};
    DecisionTree nof_cores_tree = DecisionTree(
      NOF_FEATURES_CORES_TREE, thresholds_cores_tree, indices_cores_tree, left_childs_cores_tree,
      right_childs_cores_tree, class_predictions_cores_tree);
    unsigned nof_workers = nof_cores_tree.predict(features);
    return std::min(nof_cores, nof_workers);
  }

  // Calculate the optimal C based on the problem size, config and machine parameters.
  static unsigned get_optimal_c(
    const MSMConfig& config,
    const int msm_size,
    const uint32_t scalar_size,
    const uint32_t precompute_factor,
    const uint32_t nof_workers,
    const std::string& cpu_vendor)
  {
    if (config.c > 0) { return config.c; }

    // Use decision tree to predict the optimal c
    double field_size_to_fixed_size_ratio = scalar_size / FIXED_SCALAR_SIZE_C_TREE;
    double pcm = (double)precompute_factor;
    double msm_log_size = (double)std::log2(msm_size * field_size_to_fixed_size_ratio);
    double nof_cores = (double)nof_workers;
    double features[NOF_FEATURES_C_TREE] = {msm_log_size, nof_cores, pcm};
    unsigned optimal_c;
    if (std::is_same_v<A, affine_t>) {
      if (cpu_vendor == "ARM") {
        DecisionTree arm_tree = DecisionTree(
          NOF_FEATURES_C_TREE_ARM, thresholds_c_tree_arm, indices_c_tree_arm, left_childs_c_tree_arm,
          right_childs_c_tree_arm, class_predictions_c_tree_arm);
        optimal_c = arm_tree.predict(features);
      } else if (cpu_vendor == "Intel") {
        DecisionTree intel_tree = DecisionTree(
          NOF_FEATURES_C_TREE_INTEL, thresholds_c_tree_intel, indices_c_tree_intel, left_childs_c_tree_intel,
          right_childs_c_tree_intel, class_predictions_c_tree_intel);
        optimal_c = intel_tree.predict(features);
      } else { // AMD
        DecisionTree amd_tree = DecisionTree(
          NOF_FEATURES_C_TREE_AMD, thresholds_c_tree_amd, indices_c_tree_amd, left_childs_c_tree_amd,
          right_childs_c_tree_amd, class_predictions_c_tree_amd);
        optimal_c = amd_tree.predict(features);
      }
    }
#ifdef G2_ENABLED
    else if (std::is_same_v<A, g2_affine_t>) {
      if (cpu_vendor == "ARM") {
        DecisionTree arm_tree = DecisionTree(
          NOF_FEATURES_C_TREE_ARM_G2, thresholds_c_tree_arm_g2, indices_c_tree_arm_g2, left_childs_c_tree_arm_g2,
          right_childs_c_tree_arm_g2, class_predictions_c_tree_arm_g2);
        optimal_c = arm_tree.predict(features);
      } else if (cpu_vendor == "Intel") {
        DecisionTree intel_tree = DecisionTree(
          NOF_FEATURES_C_TREE_INTEL_G2, thresholds_c_tree_intel_g2, indices_c_tree_intel_g2,
          left_childs_c_tree_intel_g2, right_childs_c_tree_intel_g2, class_predictions_c_tree_intel_g2);
        optimal_c = intel_tree.predict(features);
      } else { // AMD
        DecisionTree amd_tree = DecisionTree(
          NOF_FEATURES_C_TREE_AMD_G2, thresholds_c_tree_amd_g2, indices_c_tree_amd_g2, left_childs_c_tree_amd_g2,
          right_childs_c_tree_amd_g2, class_predictions_c_tree_amd_g2);
        optimal_c = amd_tree.predict(features);
      }
    }
#endif
    return optimal_c;
  }

private:
  // A single bucket data base
  // TBD: check adding a pointer to affine to reduce copies
  struct Bucket {
    P point;
  };

  // A single segment data base
  struct Segment {
    P line_sum;
    P triangle_sum;
  };

  // members
  tf::Taskflow m_taskflow;       // Accumulate tasks
  tf::Executor m_executor;       // execute all tasks accumulated on multiple threads
  const int m_msm_size;          // number of scalars in the problem
  const MSMConfig& m_config;     // extra parameters for the problem
  uint32_t m_scalar_size;        // the number of bits at the scalar
  uint32_t m_c;                  // the number of bits each bucket module is responsible for
  uint32_t m_bm_size;            // number of buckets in a single bucket module.
  uint32_t m_nof_buckets_module; // number of bucket modules. Each BM contains m_bm_size buckets except for the last one
  uint64_t m_nof_total_buckets;  // total number of buckets across all bucket modules
  uint32_t m_precompute_factor;  // the number of bases precomputed for each scalar
  uint32_t m_segment_size;       // segments size for phase 2.
  uint32_t m_nof_workers;        // number of threads in current machine
  std::string m_cpu_vendor;      // CPU vendor
  std::unique_ptr<std::thread> m_phase3_thread = nullptr;

  // per worker:
  std::vector<std::vector<Bucket>> m_workers_buckets;    // all buckets used by the worker
  std::vector<std::vector<bool>> m_workers_buckets_busy; // for each bucket, an indication if it is busy

  std::vector<Segment> m_segments; // A vector of all segments for phase 2

  // set the parameters based on the problem size and the machine properties
  void calc_optimal_parameters()
  {
    m_precompute_factor = m_config.precompute_factor;
    m_scalar_size = scalar_t::NBITS; // TBD handle this config.bitsize != 0 ? config.bitsize : scalar_t::NBITS;
    m_nof_workers = get_optimal_nof_workers(m_config, m_msm_size, m_scalar_size, m_precompute_factor);

    // phase 1 properties
    m_c = get_optimal_c(m_config, m_msm_size, m_scalar_size, m_precompute_factor, m_nof_workers, m_cpu_vendor);

    m_nof_buckets_module = ((m_scalar_size - 1) / (m_config.precompute_factor * m_c)) + 1;
    m_bm_size = 1 << (m_c - 1);
    const uint64_t last_bm_size =
      m_precompute_factor > 1 ? m_bm_size : 1 << (m_scalar_size - ((m_nof_buckets_module - 1) * m_c));
    m_nof_total_buckets = (m_nof_buckets_module - 1) * m_bm_size + last_bm_size;

    // phase 2 properties
    m_segment_size =
      std::min(m_bm_size, (uint32_t)(1 << (uint32_t)(std::log2(m_nof_total_buckets / m_nof_workers) - 4)));
    const int nof_segments = (m_nof_total_buckets + m_segment_size - 1) / m_segment_size;
    m_segments.resize(nof_segments);
  }

  // execute all tasks in taskfkow, wait for them to complete and clear taskflow.
  void run_workers_and_wait()
  {
    m_executor.run(m_taskflow).wait();
    m_taskflow.clear();
  }

  // if there is a phase3 task runninf, wait for it to complete
  void wait_last_task_completed()
  {
    if (m_phase3_thread) { m_phase3_thread->join(); }
  }

  // phase 1: Each worker process a portion of the inputs and populate its buckets
  void phase1_populate_buckets(const scalar_t* scalars, const A* bases)
  {
    // Divide the msm problem to workers
    const int worker_msm_size = (m_msm_size + m_nof_workers - 1) / m_nof_workers; // round up

    // Run workers to build their buckets on a subset of the scalars and bases
    for (int worker_i = 0; worker_i < m_nof_workers; worker_i++) {
      m_taskflow.emplace([&, worker_i]() {
        const int scalars_start_idx = worker_msm_size * worker_i;
        const int bases_start_idx = scalars_start_idx * m_precompute_factor;
        const int cur_worker_msm_size = std::min(worker_msm_size, m_msm_size - worker_i * worker_msm_size);
        worker_run_phase1(worker_i, scalars + scalars_start_idx, bases + bases_start_idx, cur_worker_msm_size);
      });
    }

    // TBD build a graph of dependencies for task flow to execute phase 2
    run_workers_and_wait();
  }

  // Each worker run this function and update its buckets - this function needs re writing
  void worker_run_phase1(const int worker_idx, const scalar_t* scalars, const A* bases, const int msm_size)
  {
    // Init My buckets:
    std::vector<Bucket>& buckets = m_workers_buckets[worker_idx];
    std::vector<bool>& buckets_busy = m_workers_buckets_busy[worker_idx];
    fill(buckets_busy.begin(), buckets_busy.end(), false);

    const int coeff_bit_mask_no_sign_bit = m_bm_size - 1;
    const int coeff_bit_mask_with_sign_bit = (1 << m_c) - 1;
    // NUmber of windows / additions per scalar in case num_bms * precompute_factor exceed scalar width
    const int num_bms_before_precompute = ((m_scalar_size - 1) / m_c) + 1; // +1 for ceiling
    int carry = 0;
    for (int i = 0; i < msm_size; i++) {
      carry = 0;
      // Handle required preprocess of scalar
      scalar_t scalar =
        m_config.are_scalars_montgomery_form ? scalar_t::from_montgomery(scalars[i]) : scalars[i]; // TBD: avoid copy
      bool negate_p_and_s = scalar.get_scalar_digit(m_scalar_size - 1, 1) > 0;
      if (negate_p_and_s) { scalar = scalar_t::neg(scalar); } // TBD: inplace

      for (int j = 0; j < m_precompute_factor; j++) {
        // Handle required preprocess of base P. Note: no need to convert to montgomery. Projective point handles it)
        const A& base = bases[m_precompute_factor * i + j];
        if (base == A::zero()) { continue; } // TBD: why is that? can be done more efficiently?
        const A base_neg = A::neg(base);

        for (int bm_i = 0; bm_i < m_nof_buckets_module; bm_i++) {
          // Avoid seg fault in case precompute_factor*c exceeds the scalar width by comparing index with num additions
          if (m_nof_buckets_module * j + bm_i >= num_bms_before_precompute) { break; }

          uint32_t curr_coeff = scalar.get_scalar_digit(m_nof_buckets_module * j + bm_i, m_c) + carry;

          // For the edge case of curr_coeff = c (limb=c-1, carry=1) use the sign bit mask
          if ((curr_coeff & coeff_bit_mask_with_sign_bit) != 0) {
            // Remove sign to infer the bkt idx.
            carry = curr_coeff > m_bm_size;
            int bkt_idx = carry ? m_bm_size * bm_i + ((-curr_coeff) & coeff_bit_mask_no_sign_bit)
                                : m_bm_size * bm_i + (curr_coeff & coeff_bit_mask_no_sign_bit);

            // Check for collision in that bucket and either dispatch an addition or store the P accordingly.
            if (buckets_busy[bkt_idx]) {
              buckets[bkt_idx].point =
                buckets[bkt_idx].point + ((negate_p_and_s ^ (carry > 0)) ? base_neg : base); // TBD: inplace
            } else {
              buckets_busy[bkt_idx] = true;
              buckets[bkt_idx].point =
                P::from_affine(((negate_p_and_s ^ (carry > 0)) ? base_neg : base)); // TBD: inplace
            }
          } else {
            // Handle edge case where coeff = 1 << c due to carry overflow which means:
            // coeff & coeff_mask == 0 but there is a carry to propagate to the next segment
            carry = curr_coeff >> m_c;
          }
        }
      }
    }
  }

  //   // phase 2: accumulate m_segment_size buckets into a line_sum and triangle_sum
  //   void phase2_collapse_segments()
  //   {
  //     for (int worker_i = 0; worker_i < 10;
  //       worker_i++) { // TBD: divide the work among m_nof_workers only.
  //       // Each thread is responsible for a sinעle thread
  //       m_taskflow.emplace([&, worker_i]() {
  //         for (int segment_idx = worker_i*16; segment_idx < worker_i*16+16; segment_idx++) {
  //         const uint64_t bucket_start = segment_idx * m_segment_size;
  //         const uint32_t segment_size = std::min(m_nof_total_buckets - bucket_start, (uint64_t)m_segment_size);
  //         worker_collapse_segment(m_segments[segment_idx], bucket_start, segment_size);
  //         }
  //       });
  //     }
  //     run_workers_and_wait();
  //   }

  // phase 2: accumulate m_segment_size buckets into a line_sum and triangle_sum
  void phase2_collapse_segments()
  {
    for (int segment_idx = 0; segment_idx < m_segments.size(); segment_idx++) {
      // Each thread is responsible for a sinעle thread
      m_taskflow.emplace([&, segment_idx]() {
        const uint64_t bucket_start = segment_idx * m_segment_size;
        const uint32_t segment_size = std::min(m_nof_total_buckets - bucket_start, (uint64_t)m_segment_size);
        worker_collapse_segment(m_segments[segment_idx], bucket_start, segment_size);
      });
    }
    run_workers_and_wait();
  }

  // single worker task - accumulate a single segment
  void worker_collapse_segment(Segment& segment, const int64_t bucket_start, const uint32_t segment_size)
  {
    const int64_t last_bucket_i = bucket_start + segment_size;
    const int64_t init_bucket_i = (last_bucket_i % m_bm_size) ? last_bucket_i
                                                              : // bucket 0 at the BM contains the last element
                                    last_bucket_i - m_bm_size;

    // Initialize segment sums to the first bucket
    segment.triangle_sum = P::zero();
    if (init_bucket_i < m_workers_buckets[0].size()) {
      accumulate_all_workers_buckets(init_bucket_i, segment.triangle_sum);
    }
    segment.line_sum = segment.triangle_sum;

    // run over the buckets and accumulate the to the segment
    for (int64_t bucket_i = last_bucket_i - 1; bucket_i > bucket_start; bucket_i--) {
      // add busy buckets to line sum
      accumulate_all_workers_buckets(bucket_i, segment.line_sum);
      // Add line_sum to triangle_sum
      segment.triangle_sum = segment.triangle_sum + segment.line_sum; // TBD: inplace
    }
  }

  // run over all workers and sum their bucket_idx to sum
  void accumulate_all_workers_buckets(const uint64_t bucket_idx, P& sum)
  {
    for (int worker_i = 0; worker_i < m_nof_workers; worker_i++) {
      if (m_workers_buckets_busy[worker_i][bucket_idx]) {
        sum = sum + m_workers_buckets[worker_i][bucket_idx].point; // TBD: inplace
      }
    }
  }

  // Serial phase - accumulate all segments to final result
  void phase3_final_accumulator(P* result)
  {
    const int nof_segments_per_bm = m_bm_size / m_segment_size;
    int nof_segments_left = m_segments.size();
    for (int i = 0; i < m_nof_buckets_module; i++) {
      const int cur_bm_nof_segments = std::min(nof_segments_per_bm, nof_segments_left);
      const int log_nof_segments_per_bm =
        std::log2(nof_segments_per_bm); // TBD: avoid logn. can be calculated differently.
      // Weighted sum of all the lines for each bm - summed in a similar fashion of the triangle sum of phase 2
      P partial_sum = nof_segments_per_bm * (i + 1) - 1 < m_segments.size()
                        ? m_segments[nof_segments_per_bm * (i + 1) - 1].line_sum
                        : P::zero();
      P total_sum = partial_sum;

      // run over all segments in the BM
      for (int j = cur_bm_nof_segments - 2; j > 0; j--) {
        // accumulate the partial sum
        partial_sum = partial_sum + m_segments[nof_segments_per_bm * i + j].line_sum;
        // add the partial sum to the total sum
        total_sum = total_sum + partial_sum;
      }
      m_segments[nof_segments_per_bm * i].line_sum = total_sum;

      // Convert weighted lines sum to rectangles sum by doubling
      int num_doubles = m_c - 1 - log_nof_segments_per_bm;
      for (int k = 0; k < num_doubles; k++) {
        m_segments[nof_segments_per_bm * i].line_sum = P::dbl(m_segments[nof_segments_per_bm * i].line_sum);
      }

      // Sum triangles within bm linearly
      for (int j = 1; j < cur_bm_nof_segments && nof_segments_per_bm * i + j < m_segments.size(); j++) {
        m_segments[nof_segments_per_bm * i].triangle_sum =
          m_segments[nof_segments_per_bm * i].triangle_sum + m_segments[nof_segments_per_bm * i + j].triangle_sum;
      }

      // After which add the lines and triangle sums to one sum of the entire BM
      if (cur_bm_nof_segments > 1) {
        m_segments[nof_segments_per_bm * i].triangle_sum =
          m_segments[nof_segments_per_bm * i].triangle_sum + m_segments[nof_segments_per_bm * i].line_sum;
      }
      nof_segments_left -= nof_segments_per_bm;
    }

    // Sum BM sums together
    *result = m_segments[(m_nof_buckets_module - 1) * nof_segments_per_bm].triangle_sum;
    for (int i = m_nof_buckets_module - 2; i >= 0; i--) {
      // Multiply by the BM digit factor 2^c - i.e. c doublings
      for (int j = 0; j < m_c; j++) {
        *result = P::dbl(*result);
      }
      *result = *result + m_segments[nof_segments_per_bm * i].triangle_sum;
    }
  }
}; // end of class Msm

/**
 * @brief Super function that handles the Msm class to calculate a MSM.
 * @param device - Icicle API parameter stating the device being ran on. In this case - CPU.
 * @param scalars - Input scalars for MSM.
 * @param bases - EC P input, affine representation.
 * @param msm_size - Size of the above arrays, as they are given as Pers.
 * @param config - configuration containing parameters for the MSM.
 * @param results - Per to P array in which to store the results. NOTE: the user is expected to preallocate the
 *                  results array.
 */
template <typename A, typename P>
eIcicleError cpu_msm(
  const Device& device, const scalar_t* scalars, const A* bases, int msm_size, const MSMConfig& config, P* results)
{
  Msm<A, P> msm(msm_size, config);
  for (int batch_i = 0; batch_i < config.batch_size; batch_i++) {
    const int batch_start_idx = msm_size * batch_i;
    const int bases_start_idx = config.are_points_shared_in_batch ? 0 : batch_start_idx;
    msm.run_msm(
      &scalars[batch_start_idx], &bases[bases_start_idx], &results[batch_i], batch_i + 1 == config.batch_size);
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief Function to precompute basess multiplications - trading memory for MSM performance.
 * @param device - Icicle API parameter stating the device being ran on. In this case - CPU.
 * @param input_bases - bases (EC points) to precompute.
 * @param nof_bases - Size of the above array, as it is given as a Per.
 * @param config - configuration containing parameters for the MSM. In this case, the config implicitly determines the
 *                 multiplication factor(s) of the input bases.
 * @param output_bases - Per to P array in which to store the results. NOTE: the user is expected to
 *                       preallocate the results array.
 */
template <typename A, typename P>
eIcicleError cpu_msm_precompute_bases(
  const Device& device,
  const A* input_bases,
  int nof_bases,
  const MSMConfig& config,
  A* output_bases) // Pre assigned?
{
  const int precompute_factor = config.precompute_factor;
  const uint scalar_size = scalar_t::NBITS; // TBD handle this config.bitsize != 0 ? config.bitsize : scalar_t::NBITS;
  const std::string cpu_vendor = get_cpu_vendor();
  const uint32_t nof_workers = Msm<A, P>::get_optimal_nof_workers(config, nof_bases, scalar_size, precompute_factor);
  const int c = Msm<A, P>::get_optimal_c(config, nof_bases, scalar_size, precompute_factor, nof_workers, cpu_vendor);
  const bool is_mont = config.are_points_montgomery_form;
  const unsigned int num_bms_no_precomp = (scalar_size - 1) / c + 1;
  const unsigned int shift = c * ((num_bms_no_precomp - 1) / precompute_factor + 1);
  for (int i = 0; i < nof_bases; i++) {
    output_bases[precompute_factor * i] = input_bases[i];
    P point = P::from_affine(is_mont ? A::from_montgomery(input_bases[i]) : input_bases[i]);
    for (int j = 1; j < precompute_factor; j++) { // TBD parallelize this
      for (int k = 0; k < shift; k++) {
        point = P::dbl(point);
      }
      output_bases[precompute_factor * i + j] = is_mont ? A::to_montgomery(P::to_affine(point)) : P::to_affine(point);
    }
  }
  return eIcicleError::SUCCESS;
}
