#pragma once


#include <thread>
#include <atomic>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/config_extension.h"
#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"
#include "icicle/msm.h"
#include "tasks_manager.h"
#include "timer.cpp"

using namespace icicle;
#ifdef DUMMY_TYPES // for testing
  using affine_t = DummyPoint;
  using projective_t = DummyPoint;
  using scalar_t = DummyScalar;
#else
  using affine_t = curve_config::affine_t;
  using projective_t = curve_config::projective_t;
  using scalar_t = curve_config::scalar_t;
#endif

/**
 * @class EcAddTask
 * @brief class deriving `TaskBase` to allow use of tasks_manager.h for MSM
 * (The task to be executed is an elliptic curve addition).
 */
template<typename Point>
class EcAddTask : public TaskBase
{
public:
  /**
   * @brief constructor for the task ensuring points are zeros and other fields are init to invalid values (to avoid 
   * falsely handling a result at the start). The execution adds points p1,p2 and stores the result in p1.
   */
  EcAddTask() : TaskBase(), p1(Point::zero()), p2(Point::zero()), return_idx(-1) {}
  void execute() { p1 = p1 + p2; } // COMMENT execute many for AVX
  
  Point p1; // TODO allow copy by reference as well (phase1 still needs copy by value)
  Point p2;
  int return_idx;     // Idx allowing manager to figure out where the result belong to.
  bool is_line=false; // Indicator for phase 2 sums between line sum and triangle sum.
};

/**
 * @class MSM
 * @brief class for solving multi-scalar-multiplication on the cpu.
 * The class is a template depending on the element relevant to MSM (for instance EC point).
 */
template <typename Point>
class Msm
{
public:
  /**
   * @brief Constructor for Msm
   * @param config - msm config. important parameters that are part of the config extension are: n_threads, c
   */
  Msm(const MSMConfig& config);

  ~Msm() { if (p3_thread.joinable()) { std::cout << "\n\nYou didn't join me!\n\n\n"; p3_thread.join(); }}

  /**
   * @brief Phase 1 (accumulation) of MSM
   * @param scalars - scalar input.
   * @param bases - EC point input, affine representation (Regardless of the defined Point type of the class).
   * @param msm_size - Size of the above arrays, as they are given as pointers.
   */
  void bucket_accumulator(
    const scalar_t* scalars,
    const affine_t* bases,
    const unsigned int msm_size);

  /**
   * @brief Function handling the initial parallel part of summing the bucket modules. It splits the BMs into segments, 
   * summing each in separate and passing the segments sum to be handled by the final accumulator.
   * @param is_first_in_batch - flag indicating this is the first MSM in the batch, therefore there is no need in 
   * checking and blocking if the previous (Non-existent) MSM phase 3 has finished,
   */
  void bm_sum(bool is_first_in_batch);

  /**
   * @brief Final accumulation required for MSM calculation. the function will either launch a thread to perform the 
   * calculation (`phase3_tread` function) or by the main thread, depending on the position in the batch (Last msm or 
   * not).
   * @param result - pointer to write the MSM result to. Memory for the pointer has already been allocated by the user.
   * @param is_last_in_batch - flag indicating this is the last MSM in the batch, therefore there is no need in 
   * launching another thread while the main starts a new MSM.
   */
  void final_accumulator(Point* result, bool is_last_in_batch);

  /**
   * @brief Function for resetting class members between batch runs.
   */
  void batch_run_reset();

private:
  TasksManager<EcAddTask<Point>> manager; // Tasks manager for multithreading

  const unsigned int m_c;                 // Pipenger's constant
  const unsigned int m_num_bkts;          // Number of buckets in each bucket module
  const unsigned int m_precompute_factor; // multiplication of points already calculated trading memory for performance
  const unsigned int m_num_bms;           // Number of bucket modules (windows in Pipenger's algorithm)
  const bool m_are_scalars_mont;          // Are the input scalars in Montgomery representation
  const bool m_are_points_mont;           //  Are the input points in Montgomery representation

  // Phase 1 members
  std::vector<Point> m_buckets;           // Vector of all buckets required for phase 1 (All bms in order)
  std::vector<bool> m_bkts_occupancy;     // Boolean vector indicating if the corresponding bucket is occupied

  // Phase 2 members
  const int m_log_num_segments;
  const int m_num_bm_segments;
  const int m_segment_size;

  /**
   * @struct BmSumSegment
   * @brief Struct bundling the required data members for each BM segment in phase 2.
   */
  struct BmSumSegment
  {
    Point triangle_sum;
    Point line_sum;
    
    int num_received_sums = 1;  // Counter for numbers of sum received for this row idx - used to determine when a new 
                                // row idx can be dispatched (Due to it depending on the two sums from the prev idx).
    int row_to_sum_idx = -1;    // Counter counting down how far the current sum is through the segment.
    int segment_mem_offset = -1;// Offset in 'm_buckets' of this segment.
  };
  
  std::vector<BmSumSegment> m_segments;

  // Phase 3 members
  std::thread p3_thread;

  /**
   * @brief Push addition task during phase 1.
   * Th function also handles completed addition results while attempting to insert the new addition task (including 
   * potential collision that requires more urgent ec-addition of the result and the current stored value).
   * @param task_bkt_idx - address in m_buckets in which to store the result in the future.
   * @param bkt - the point from m_buckets.
   * @param base - the point from the input bases
   */
  void phase1_push_addition(const unsigned int task_bkt_idx, const Point bkt, const Point& base);
  /**
   * @brief Handles the final results of phase 1 (after no other planned additions are required).
   * The function also handles the potential collision similarly to push_addition above.
   */
  void phase1_wait_for_completion();

  /**
   * @brief Setting up phase 2 class members according to phase 1 results. 
   * NOTE: This setup function also potentially blocks until the previous MSM's phase 3 finishes (Given that in batched 
   * MSM, phase 3 is initiated on a separate thread while the next MSM begins phase 1).
   */
  void phase2_setup(bool is_first_in_batch);

  /**
   * @brief Phase 3 function to be ran by a separate thread (or main in the end of the run) - i.e. without using the 
   * tasks manager. It performs the remaining serial addition in each BM and sums them to one final MSM result.
   * @param result - pointer to write the MSM result to. Memory for the pointer has already been allocated by the user.
   */
  void phase3_thread(Point* result);
};

// COMMENT The single threaded functions will be replaced in the future by multithreaded counter parts

template <typename Point>
Msm<Point>::Msm(const MSMConfig& config)
    : manager(config.ext->get<int>("n_threads") > 0? config.ext->get<int>("n_threads") :
        std::thread::hardware_concurrency()),

      m_c(config.ext->get<int>("c")),
      m_num_bkts(1 << (m_c - 1)),
      m_precompute_factor(config.precompute_factor), 
      m_num_bms(((scalar_t::NBITS - 1) / (config.precompute_factor * m_c)) + 1),
      m_are_scalars_mont(config.are_scalars_montgomery_form), 
      m_are_points_mont(config.are_points_montgomery_form),

      m_buckets(m_num_bms * m_num_bkts, Point::zero()), 
      m_bkts_occupancy(m_num_bms * m_num_bkts, false),

      m_log_num_segments(std::max((int)std::floor(
        std::log2(config.ext->get<int>("n_threads") * TASKS_PER_THREAD / (2 * m_num_bms))), 0)),
      m_num_bm_segments(std::min((int)(1 << m_log_num_segments), (int)(m_num_bms*m_num_bkts))), 
      m_segment_size(std::max((int)(m_num_bkts >> m_log_num_segments), 1)), 
      m_segments(m_num_bms * m_num_bm_segments)
{}

template <typename Point>
void Msm<Point>::phase1_wait_for_completion()
{
  EcAddTask<Point>* task = manager.get_completed_task();
  while (task != nullptr)
  {
    if (m_bkts_occupancy[task->return_idx])
    {
      task->p2 = m_buckets[task->return_idx];
      m_bkts_occupancy[task->return_idx] = false;
      task->dispatch();
    }
    else
    {
      m_buckets[task->return_idx] = task->p1;
      m_bkts_occupancy[task->return_idx] = true;
      task->set_idle();
    }
    task = manager.get_completed_task();
  }
}

template <typename Point>
void Msm<Point>::phase1_push_addition(
  const unsigned int task_bkt_idx,
  const Point bkt,
  const Point& base) // TODO add option of adding different types
{
  /**
   * Assign EC addition to a free thread
   * @param task_bkt_idx - result's address in the bkts array
   * @param bkt - bkt to be added. it is passed by value to allow the appropriate cell in the bucket array to be "free"
   * an overwritten without affecting the working thread
   * @param p2 - point to be added
   */
  EcAddTask<Point>* task = nullptr;
  while (task == nullptr) 
  {
    task = manager.get_idle_or_completed_task();
    if (task->is_completed())
    {
      if (m_bkts_occupancy[task->return_idx])
      {
        task->p2 = m_buckets[task->return_idx];
        m_bkts_occupancy[task->return_idx] = false;
        task->dispatch();
        task = nullptr;
        continue;
      }
      else 
      {
        m_buckets[task->return_idx] = task->p1;
        m_bkts_occupancy[task->return_idx] = true;
      }
    }
    task->p1 = bkt;
    task->p2 = base;
    task->return_idx = task_bkt_idx;
    task->dispatch();
  }
}

template <typename Point>
void Msm<Point>::bucket_accumulator(const scalar_t* scalars, const affine_t* bases, const unsigned int msm_size)
{
  auto t = Timer("P1:bucket-accumulator"); // TODO remove all timers
  const int coeff_bit_mask_no_sign_bit = m_num_bkts - 1;
  const int coeff_bit_mask_with_sign_bit = (1 << m_c) - 1;
  // NUmber of windows / additions per scalar in case num_bms * precompute_factor exceed scalar width
  const int num_additions_per_scalar = (scalar_t::NBITS - 1) / m_c; // +1 for ceiling than -1 for m1

  int carry = 0;
  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    // TODO multithreaded montgomery once the new TasksManager multi task api is available
    scalar_t scalar = m_are_scalars_mont ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(scalar_t::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = scalar_t::neg(scalar);
    for (int j = 0; j < m_precompute_factor; j++) {
      affine_t point = m_are_points_mont ? affine_t::from_montgomery(bases[m_precompute_factor * i + j])
                                       : bases[m_precompute_factor * i + j];
      if (negate_p_and_s) point = affine_t::neg(point);
      for (int k = 0; k < m_num_bms; k++) {
        // Avoid seg fault in case precompute_factor*c exceeds the scalar width by comparing index with num additions
        if (m_num_bms * j + k > num_additions_per_scalar) { break; }

        uint32_t curr_coeff = scalar.get_scalar_digit(m_num_bms * j + k, m_c) + carry;
        int bkt_idx = 0;
        if ((curr_coeff & coeff_bit_mask_with_sign_bit) != 0)
        {
          // Remove sign to infer the bkt idx.
          carry = curr_coeff >= m_num_bkts;
          if (curr_coeff < m_num_bkts) { bkt_idx = m_num_bkts * k + curr_coeff; } 
          else { bkt_idx = m_num_bkts * k + ((-curr_coeff) & coeff_bit_mask_no_sign_bit); }
          // Check for collision in that bucket and either dispatch an addition or store the point accordingly.
          if (m_bkts_occupancy[bkt_idx]) {
            m_bkts_occupancy[bkt_idx] = false;
            phase1_push_addition(bkt_idx, m_buckets[bkt_idx], carry > 0 ? Point::from_affine(affine_t::neg(point)) : 
                                  Point::from_affine(point));
          } else {
            m_bkts_occupancy[bkt_idx] = true;
            m_buckets[bkt_idx] = carry > 0 ? Point::neg(Point::from_affine(point)) : Point::from_affine(point);
          }
        } else {
          // Handle edge case where coeff = 1 << c due to carry overflow which means: 
          // coeff & coeff_mask == 0 but there is a carry to propagate to the next segment
          carry = curr_coeff >> m_c;
        }
      }
    }
  }
  phase1_wait_for_completion();
}

template <typename Point>
void Msm<Point>::phase2_setup(bool is_first_in_batch)
{ 
  // For batched msm ensuring phase3 finishes before next phase2 starts and overrides data
  if (!is_first_in_batch) { p3_thread.join(); }

  // Init values of partial (line) and total (triangle) sum
  for (int i = 0; i < m_num_bms; i++) {
    for (int j = 0; j < m_num_bm_segments - 1; j++) {
      m_segments[m_num_bm_segments * i + j].triangle_sum =   m_buckets[m_num_bkts * i + m_segment_size * (j + 1)];
      m_segments[m_num_bm_segments * i + j].line_sum =  m_buckets[m_num_bkts * i + m_segment_size * (j + 1)];
      m_segments[m_num_bm_segments * i + j].row_to_sum_idx = m_segment_size - 2;
      m_segments[m_num_bm_segments * i + j].segment_mem_offset = m_num_bkts * i + m_segment_size * j + 1;
    }
    // The most significant bucket of every bm is stored in address 0 - 
    // so the last tri/line sums will be initialized to bucket[0]
    m_segments[m_num_bm_segments * (i + 1) - 1].triangle_sum =   m_buckets[m_num_bkts * i];
    m_segments[m_num_bm_segments * (i + 1) - 1].line_sum =  m_buckets[m_num_bkts * i];
    m_segments[m_num_bm_segments * (i + 1) - 1].row_to_sum_idx = m_segment_size - 2;
    m_segments[m_num_bm_segments * (i + 1) - 1].segment_mem_offset = m_num_bkts * (i + 1) - m_segment_size + 1;
  }
}

template <typename Point>
void Msm<Point>::bm_sum(bool is_first_in_batch)
{
  auto t = Timer("P2:bm-sums");
  phase2_setup(is_first_in_batch);
  
  if (m_segment_size > 1)
  {
    // Send first additions - line additions
    for (int i = 0; i < m_num_bms; i++)
    {
      for (int j = 0; j < m_num_bm_segments; j++)
      {
        EcAddTask<Point>* task = manager.get_idle_or_completed_task();
        BmSumSegment& curr_segment = m_segments[m_num_bm_segments * i + j]; // For readability

        task->p1 = curr_segment.line_sum;
        task->p2 = m_buckets[curr_segment.segment_mem_offset + curr_segment.row_to_sum_idx];
        task->return_idx = m_num_bm_segments * i + j;
        task->is_line = true;

        task->dispatch();
      }
    }

    // Loop until all line/tri sums are done
    int done_segments = 0;
    while (done_segments < m_num_bms * m_num_bm_segments)
    {
      EcAddTask<Point>* task = manager.get_completed_task();
      BmSumSegment& curr_segment = m_segments[task->return_idx]; // For readability
      
      // Handle result and check if it is the last
      if (task->is_line) { curr_segment.line_sum = task->p1; } else { curr_segment.triangle_sum = task->p1; }
      curr_segment.num_received_sums++;
      
      if (curr_segment.row_to_sum_idx < 0)
      {
        done_segments++;
        task->set_idle();
        continue;
      }
      // Otherwise check if it is possible to assign new additions:
      // Triangle sum is dependent on the 2 previous sums (line and triangle) - so check if 2 sums were received.
      if (curr_segment.num_received_sums == 2)
      {
        // Triangle sum
        curr_segment.num_received_sums = 0;
        task->p1 = curr_segment.line_sum;
        task->p2 = curr_segment.triangle_sum;
        task->is_line = false;
        
        task->dispatch();
        curr_segment.row_to_sum_idx--;

        // Line sum (if not the last one in the segment)
        // NOTE doesn't allow passing point by reference to EcAddTask as curr_segment.line_sum serving as input to both 
        // triangle and line sum can be modified by line sum.
        if (curr_segment.row_to_sum_idx >= 0) 
        { 
          int prev_idx = task->return_idx;
          task = manager.get_idle_task(); 

          task->return_idx = prev_idx;
          task->p1 = curr_segment.line_sum;
          task->p2 = m_buckets[curr_segment.segment_mem_offset + curr_segment.row_to_sum_idx];
          task->is_line = true;
          task->dispatch();
        }
      }
      else { task->set_idle(); } // Handling Completed task without dispatching a new one
    }
  }
}

template <typename Point>
void Msm<Point>::phase3_thread(Point* result)
{
  for (int i = 0; i < m_num_bms; i++)
  {
    // Weighted sum of all the lines for each bm - summed in a similar fashion of the triangle sum of phase 2
    Point partial_sum = m_segments[m_num_bm_segments * (i + 1) - 1].line_sum;
    Point total_sum =   m_segments[m_num_bm_segments * (i + 1) - 1].line_sum;
    for (int j = m_num_bm_segments - 2; j > 0; j--)
    {
      partial_sum = partial_sum + m_segments[m_num_bm_segments * i + j].line_sum;
      total_sum = total_sum + partial_sum;
    } 
    m_segments[m_num_bm_segments * i].line_sum = total_sum;

    // Convert weighted lines sum to rectangles sum by doubling
    int num_doubles = m_c - 1 - m_log_num_segments;
    for (int k = 0; k < num_doubles; k++)
    {
      m_segments[m_num_bm_segments * i].line_sum =  m_segments[m_num_bm_segments * i].line_sum + 
                                                    m_segments[m_num_bm_segments * i].line_sum;
    }

    // Sum triangles within bm linearly
    for (int j = 1; j < m_num_bm_segments; j++)
    {
      m_segments[m_num_bm_segments * i].triangle_sum =  m_segments[m_num_bm_segments * i].triangle_sum + 
                                                        m_segments[m_num_bm_segments * i + j].triangle_sum;
    }
    // After which add the lines and triangle sums to one sum of the entire BM
    m_segments[m_num_bm_segments * i].triangle_sum =  m_segments[m_num_bm_segments * i].triangle_sum + 
                                                      m_segments[m_num_bm_segments * i].line_sum;
  }
  
  // Sum BM sums together
  Point final_sum = m_segments[(m_num_bms - 1) * m_num_bm_segments].triangle_sum;
  for (int i = m_num_bms - 2; i >= 0; i--)
  {
    // Multiply by the BM digit factor 2^c - i.e. c doublings
    for (int j = 0; j < m_c; j++) { final_sum = final_sum + final_sum; }
    final_sum = final_sum + m_segments[m_num_bm_segments * i].triangle_sum;
  }
  *result = final_sum;
}

template <typename Point>
void Msm<Point>::final_accumulator(Point* result, bool is_last_in_batch)
{
  if (is_last_in_batch) { phase3_thread(result); }
  else { p3_thread = std::thread(&Msm<Point>::phase3_thread, this, result); }
}

template <typename Point>
void Msm<Point>::batch_run_reset()
{
  std::fill(m_bkts_occupancy.begin(), m_bkts_occupancy.end(), false);
  std::fill(m_buckets.begin(), m_buckets.end(), Point::zero());
}

// None class functions below:

/**
 * @brief Function to check the MSM config is valid for calculating in CPU.
 * @return - status if the config is supported or not.
 */
eIcicleError not_supported(const MSMConfig& conf)
{
  if (conf.is_async) { return eIcicleError::INVALID_DEVICE; }
  // There is only host therefore the following configs are not supported:
  if (conf.are_scalars_on_device | conf.are_points_on_device | conf.are_results_on_device)
    { return eIcicleError::INVALID_DEVICE; }
  return eIcicleError::SUCCESS;
}

/**
 * @brief Super function that handles the Msm class to calculate a MSM.
 * @param device - Icicle API parameter stating the device being ran on. In this case - CPU.
 * @param scalars - Input scalars for MSM.
 * @param bases - EC point input, affine representation.
 * @param msm_size - Size of the above arrays, as they are given as pointers.
 * @param config - configuration containing parameters for the MSM.
 * @param results - pointer to Point array in which to store the results. NOTE: the user is expected to preallocate the 
 *                  results array.
 */
template <typename Point>
eIcicleError cpu_msm(
  const Device& device,
  const scalar_t* scalars,
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  Point* results)
{
  Msm<Point>* msm = new Msm<Point>(config);
  auto t = Timer("total-msm");
  
  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext->get<int>("c");
  const unsigned int precompute_factor = config.precompute_factor;
  const int num_bms = ((scalar_t::NBITS - 1) / (precompute_factor * c)) + 1;

  if (config.ext->get<int>("n_threads") <= 0) { return eIcicleError::INVALID_ARGUMENT; }
  
  for (int i = 0; i < config.batch_size; i++)
  {
    msm->bucket_accumulator(&scalars[msm_size*i], bases, msm_size);
    msm->bm_sum(i == 0);
    msm->final_accumulator(&results[i], i == config.batch_size - 1);
    if (i < config.batch_size - 1) { msm->batch_run_reset(); }
  }  
  delete msm;
  return eIcicleError::SUCCESS;
}

/**
 * @brief Function to precompute points multiplications - trading memory for MSM performance.
 * @param device - Icicle API parameter stating the device being ran on. In this case - CPU.
 * @param input_bases - points to precompute.
 * @param nof_bases - Size of the above array, as it is given as a pointer.
 * @param config - configuration containing parameters for the MSM. In this case, the config implicitly determines the 
 *                 multiplication factor(s) of the input bases.
 * @param output_bases - pointer to Point array in which to store the results. NOTE: the user is expected to 
 *                       preallocate the results array.
 */
template <typename A>
eIcicleError cpu_msm_precompute_bases(
  const Device& device,
  const A* input_bases,
  int nof_bases,
  const MSMConfig& config,
  A* output_bases) // Pre assigned?
{
  int precompute_factor = config.precompute_factor;
  bool is_mont = config.are_points_montgomery_form;
  // bool is_mont=false;
  const unsigned int c = config.ext->get<int>("c");
  const unsigned int num_bms_no_precomp = (scalar_t::NBITS - 1) / c + 1;
  const unsigned int shift = c * ((num_bms_no_precomp - 1) / precompute_factor + 1);
  for (int i = 0; i < nof_bases; i++) {
    output_bases[precompute_factor * i] = input_bases[i];
    projective_t point = projective_t::from_affine(is_mont ? A::from_montgomery(input_bases[i]) : input_bases[i]);
    for (int j = 1; j < precompute_factor; j++) {
      for (int k = 0; k < shift; k++) {
        point = projective_t::dbl(point);
      }
      output_bases[precompute_factor * i + j] = is_mont
                                                  ? A::to_montgomery(projective_t::to_affine(point))
                                                  : projective_t::to_affine(point);
    }
  }
  return eIcicleError::SUCCESS;
}