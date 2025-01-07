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
#include <taskflow/taskflow.hpp>
#include "icicle/backend/msm_config.h"
#ifdef MEASURE_MSM_TIMES
  #include "icicle/utils/timer.hpp"
#endif

using namespace icicle;
using namespace curve_config;

template <typename A, typename P>
class Msm
{
public:
  // Constructor
  Msm(const int msm_size, const MSMConfig& config) : m_msm_size(msm_size), m_config(config)
  {
    // std::cout << "msm_size = " << msm_size << std::endl;
    // std::cout << "c = " << config.c << std::endl;
    // std::cout << "precompute_factor = " << config.precompute_factor << std::endl;
    // std::cout << "batch_size = " << config.batch_size << std::endl;
    calc_optimal_parameters();

    // resize the thread buckets according to the parameters
    m_workers_buckets.resize(m_nof_workers);
    m_workers_buckets_busy.resize(m_nof_workers);
    for (int worker_i = 0; worker_i < m_nof_workers; worker_i++) {
      m_workers_buckets[worker_i].resize(m_nof_total_buckets);
      m_workers_buckets_busy[worker_i].resize(m_nof_total_buckets);
    }
  }

  // run MSM
  void run_msm(const scalar_t* scalars, const A* bases, P* results)
  {
    // initialize all buckets to not busy
    init_workers_buckets_busy();

    phase1_populate_buckets(scalars, bases);
    // for (int i=0; i<m_workers_buckets[0].size();++i) {
    //   std::cout << "bkt " << i << " : " << m_workers_buckets_busy[0][i] << ", val = " <<
    //   m_workers_buckets[0][i].point.to_affine() << std::endl;
    // }

    phase2_collapse_segments();

    phase3_final_accumulator(results);
    std::cout << "Final result: " << results->to_affine() << std::endl;
  }

  // Calculate the optimal C based on the problem size, config and machine parameters.
  static unsigned get_optimal_c(unsigned msm_size, const MSMConfig& config)
  {
    // TBD: optimize - condsider batch size
    int optimal_c = config.c > 0 ? config.c : std::max((int)(0.7 * std::log2(msm_size * config.precompute_factor)), 8);
    return optimal_c;
  }

private:
  // A single bucket data base
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
  uint32_t m_segment_size;
  uint32_t m_nof_workers; // number of threads in current machine

  // per worker, per
  std::vector<std::vector<Bucket>> m_workers_buckets; // TBD check if structuring this affect performance
  std::vector<std::vector<bool>> m_workers_buckets_busy;

  std::vector<Segment> m_segments;

  // set the parameters based on the problem size and the machine properties
  void calc_optimal_parameters()
  {
    m_nof_workers = std::thread::hardware_concurrency();
    if (m_config.ext && m_config.ext->has(CpuBackendConfig::CPU_NOF_THREADS)) {
      m_nof_workers = m_config.ext->get<int>(CpuBackendConfig::CPU_NOF_THREADS);
    }
    if (m_nof_workers <= 0) {
      ICICLE_LOG_WARNING << "Unable to detect number of hardware supported threads - fixing it to 1\n";
      m_nof_workers = 1;
    }
    m_scalar_size = scalar_t::NBITS; // TBD handle this config.bitsize != 0 ? config.bitsize : scalar_t::NBITS;
    // calc optimal C
    m_c = get_optimal_c(m_msm_size, m_config);
    m_nof_buckets_module = ((m_scalar_size - 1) / (m_config.precompute_factor * m_c)) + 1;
    m_bm_size = 1 << (m_c - 1);
    m_precompute_factor = m_config.precompute_factor;
    const uint64_t last_bm_size =
      m_precompute_factor > 1 ? m_bm_size : 1 << (m_scalar_size - ((m_nof_buckets_module - 1) * m_c));
    m_nof_total_buckets = (m_nof_buckets_module - 1) * m_bm_size + last_bm_size;

    m_segment_size =
      std::min(m_bm_size, (uint32_t)(1 << (uint32_t)(std::log2(m_nof_total_buckets / m_nof_workers) - 4)));
    // std::cout << "m_segment_size = " << m_segment_size << std::endl;
    const int nof_segments = (m_nof_total_buckets + m_segment_size - 1) / m_segment_size;
    m_segments.resize(nof_segments);
  }

  // init all the workers buckets to empty
  void init_workers_buckets_busy() // TBD move to threads - note that not al threads are working
  {
    for (auto& buckets_busy : m_workers_buckets_busy) {
      fill(buckets_busy.begin(), buckets_busy.end(), false);
    }
  }

  void run_workers_and_wait()
  {
    m_executor.run(m_taskflow).wait();
    m_taskflow.clear();
  }

  // phase 1: Each worker process a portion of the inputs and populate its buckets
  void phase1_populate_buckets(const scalar_t* scalars, const A* bases)
  {
    // divide the msm problem to workers
    const int worker_msm_size = (m_msm_size + m_nof_workers - 1) / m_nof_workers; // round up

    // Run workers to build their buckets on a subset of the scalars and bases
    for (int worker_i = 0; worker_i < m_nof_workers; worker_i++) {
      m_taskflow.emplace([=]() {
        const int scalars_start_idx = worker_msm_size * worker_i;
        const int bases_start_idx = scalars_start_idx * m_precompute_factor;
        const int cur_worker_msm_size = std::min(worker_msm_size, m_msm_size - worker_i * worker_msm_size);
        if (cur_worker_msm_size > 0) { // TBD: better calc to avoid this condition or move the condition befoe launch a
                                       // thread
          worker_run_phase1(worker_i, scalars + scalars_start_idx, bases + bases_start_idx, cur_worker_msm_size);
        }
      });
    }
    run_workers_and_wait();

    // Collapse all the workers buckets into one
    collapse_all_workers_result();
  }

  // each worker run this function and update its buckets
  void worker_run_phase1(const int worker_idx, const scalar_t* scalars, const A* bases, const unsigned int msm_size)
  {
    std::vector<Bucket>& buckets = m_workers_buckets[worker_idx];
    std::vector<bool>& buckets_busy = m_workers_buckets_busy[worker_idx];

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
        // Handle required preprocess of base P
        A base = m_config.are_points_montgomery_form ? A::from_montgomery(bases[m_precompute_factor * i + j])
                                                     : bases[m_precompute_factor * i + j]; // TDB: avoid copy
        if (base == A::zero()) { continue; }                                               // TBD: why is that?
        A base_neg = A::neg(base);

        for (int bm_i = 0; bm_i < m_nof_buckets_module; bm_i++) {
          // Avoid seg fault in case precompute_factor*c exceeds the scalar width by comparing index with num additions
          if (m_nof_buckets_module * j + bm_i >= num_bms_before_precompute) { break; }

          uint32_t curr_coeff = scalar.get_scalar_digit(m_nof_buckets_module * j + bm_i, m_c) + carry;
          int bkt_idx = 0; // TBD: move inside if and change to ( = ? : )
          // For the edge case of curr_coeff = c (limb=c-1, carry=1) use the sign bit mask
          if ((curr_coeff & coeff_bit_mask_with_sign_bit) != 0) {
            // Remove sign to infer the bkt idx.
            carry = curr_coeff > m_bm_size;
            if (!carry) {
              bkt_idx = m_bm_size * bm_i + (curr_coeff & coeff_bit_mask_no_sign_bit);
            } else {
              bkt_idx = m_bm_size * bm_i + ((-curr_coeff) & coeff_bit_mask_no_sign_bit);
            }
            //  if (bkt_idx == 277) {
            //   // std::cout << "bkt_idx = " << bkt_idx << ", scalar = " << scalar <<  std::endl;
            //   std::cout << i << ": bkt_idx = " << bkt_idx << ", base = " << base << std::endl;
            //  }
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

  void collapse_all_workers_result()
  {
    int bucket_start = 0;
    const int nof_buckets_per_thread = (m_nof_total_buckets + m_nof_workers - 1) / m_nof_workers;
    for (int worker_i = 0; worker_i < m_nof_workers; worker_i++) {
      const int nof_buckets =
        std::min(nof_buckets_per_thread, (int)m_nof_total_buckets - worker_i * nof_buckets_per_thread);
      // threads.emplace_back(std::bind(&Msm::worker_collapse_all_workers_result, this, bucket_start, nof_buckets));
      if (nof_buckets > 0) { // TBD add this condition to repo
        m_taskflow.emplace(
          [this, bucket_start, nof_buckets]() { worker_collapse_all_workers_result(bucket_start, nof_buckets); });
        bucket_start += nof_buckets_per_thread;
      }
    }
    run_workers_and_wait();
  }

  void worker_collapse_all_workers_result(const int bucket_start, const int nof_buckets)
  {
    for (int bucket_i = bucket_start; bucket_i < bucket_start + nof_buckets; bucket_i++) {
      if (!m_workers_buckets_busy[0][bucket_i]) { m_workers_buckets[0][bucket_i].point = P::zero(); }

      for (int worker_i = 1; worker_i < m_nof_workers; worker_i++) {
        if (m_workers_buckets_busy[worker_i][bucket_i]) {
          m_workers_buckets[0][bucket_i].point =
            m_workers_buckets[0][bucket_i].point + m_workers_buckets[worker_i][bucket_i].point; // TBD inplace
        }
      }
    }
  }

  void phase2_collapse_segments()
  {
    uint64_t bucket_start = 0;
    for (int segment_idx = 0; segment_idx < m_segments.size(); segment_idx++) {
      m_taskflow.emplace([=]() {
        const uint32_t segment_size = std::min(m_nof_total_buckets - bucket_start, (uint64_t)m_segment_size);
        // worker_collapse_all_workers_result(bucket_start, segment_size);
        worker_collapse_segment(m_segments[segment_idx], bucket_start, segment_size);
      });
      bucket_start += m_segment_size;
    }
    run_workers_and_wait();
  }

  void worker_collapse_segment(Segment& segment, const int64_t bucket_start, const uint32_t segment_size)
  {
    // Assumption all buyckets are busy
    std::vector<Bucket>& buckets = m_workers_buckets[0];
    const int64_t last_bucket_i = bucket_start + segment_size;
    const int64_t init_bucket_i = (last_bucket_i % m_bm_size) ? last_bucket_i
                                                              : // bucket 0 at the BM contains the last element
                                    last_bucket_i - m_bm_size;
    segment.triangle_sum = init_bucket_i < buckets.size() ? buckets[init_bucket_i].point : P::zero();
    segment.line_sum = segment.triangle_sum;
    for (int64_t bucket_i = last_bucket_i - 1; bucket_i > bucket_start; bucket_i--) {
      // if (bucket_start == 608*128) {
      //   std::cout << "line = " << segment.line_sum.to_affine() << std::endl;
      //   std::cout << "line += " << bucket_i << ":" << buckets[bucket_i].point.to_affine() << std::endl;
      // }
      segment.line_sum = segment.line_sum + buckets[bucket_i].point; // TBD: inplace

      // if (bucket_start == 608*128) {
      //   std::cout << "tri = " << segment.triangle_sum.to_affine() << std::endl;
      //   std::cout << "tri += " << segment.line_sum.to_affine() << std::endl;
      // }
      segment.triangle_sum = segment.triangle_sum + segment.line_sum; // TBD: inplace
    }
  }

  void phase3_final_accumulator(P* result)
  {
    const int nof_segments_per_bm = m_bm_size / m_segment_size;
    int nof_segments_left = m_segments.size();
    for (int i = 0; i < m_nof_buckets_module; i++) {
      const int cur_bm_nof_segments = std::min(nof_segments_per_bm, nof_segments_left);
      const int log_nof_segments_per_bm = std::log2(nof_segments_per_bm); // (m_c - 1) -
      // Weighted sum of all the lines for each bm - summed in a similar fashion of the triangle sum of phase 2
      P partial_sum = nof_segments_per_bm * (i + 1) - 1 < m_segments.size()
                        ? m_segments[nof_segments_per_bm * (i + 1) - 1].line_sum
                        : P::zero();
      P total_sum = partial_sum;
      // if (i==31) {
      //   std::cout << "m_num_bm_segments=" << cur_bm_nof_segments << ", nof_segments_per_bm=" << nof_segments_per_bm
      //   << std::endl;
      // }
      for (int j = cur_bm_nof_segments - 2; j > 0; j--) {
        partial_sum = partial_sum + m_segments[nof_segments_per_bm * i + j].line_sum;
        total_sum = total_sum + partial_sum;
        // if (i==31) {
        //   std::cout << "line idx=" << nof_segments_per_bm * i + j << std::endl;
        //   std::cout << "line=" << m_segments[nof_segments_per_bm * i + j].line_sum.to_affine() << std::endl;
        //   std::cout << "line=" << m_segments[nof_segments_per_bm * i + j].line_sum << std::endl;
        //   std::cout << "partial_sum=" << partial_sum.to_affine() << std::endl;
        //   std::cout << "total_sum=" << total_sum.to_affine() << std::endl;
        // }
      }
      m_segments[nof_segments_per_bm * i].line_sum = total_sum;
      //      std::cout << "i=" << i << ", total_sum = " << total_sum.to_affine() << std::endl;

      // Convert weighted lines sum to rectangles sum by doubling
      int num_doubles = m_c - 1 - log_nof_segments_per_bm;
      for (int k = 0; k < num_doubles; k++) {
        m_segments[nof_segments_per_bm * i].line_sum = P::dbl(m_segments[nof_segments_per_bm * i].line_sum);
      }
      //      std::cout << "i=" << i << ", m_segments[cur_bm_nof_segments * i].line_sum = " <<
      //      m_segments[nof_segments_per_bm * i].line_sum.to_affine() << std::endl;

      // Sum triangles within bm linearly
      for (int j = 1; j < cur_bm_nof_segments && nof_segments_per_bm * i + j < m_segments.size(); j++) {
        m_segments[nof_segments_per_bm * i].triangle_sum =
          m_segments[nof_segments_per_bm * i].triangle_sum + m_segments[nof_segments_per_bm * i + j].triangle_sum;
      }
      //      std::cout << "i=" << i << ", m_segments[cur_bm_nof_segments * i].triangle_sum = " <<
      //      m_segments[nof_segments_per_bm * i].triangle_sum.to_affine() << std::endl;
      // After which add the lines and triangle sums to one sum of the entire BM
      if (cur_bm_nof_segments > 1) {
        // std::cout << "i=" << i << ", adding = " << m_segments[nof_segments_per_bm * i].line_sum.to_affine() <<
        // std::endl;
        m_segments[nof_segments_per_bm * i].triangle_sum =
          m_segments[nof_segments_per_bm * i].triangle_sum + m_segments[nof_segments_per_bm * i].line_sum;
      }
      nof_segments_left -= nof_segments_per_bm;
      // std::cout << "i=" << i << ", m_segments[cur_bm_nof_segments * i].triangle_sum = " <<
      // m_segments[nof_segments_per_bm * i].triangle_sum.to_affine() << std::endl;
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
    msm.run_msm(&scalars[batch_start_idx], &bases[bases_start_idx], &results[batch_i]);
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
  int c = Msm<A, P>::get_optimal_c(nof_bases, config);

  const int precompute_factor = config.precompute_factor;
  const bool is_mont = config.are_points_montgomery_form;
  const uint scalar_size = scalar_t::NBITS; // TBD handle this config.bitsize != 0 ? config.bitsize : scalar_t::NBITS;
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
