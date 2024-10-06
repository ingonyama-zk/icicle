#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <iostream>
#include <fstream>

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/config_extension.h"
#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"
#include "icicle/msm.h"
#include "tasks_manager.h"
#include "icicle/backend/msm_config.h"
#ifdef LOG_UTILIZATION
#include "icicle/utils/timer.hpp"
#endif

using namespace icicle;
using namespace curve_config;

#define LOG_EC_BATCH_SIZE 2
#define EC_BATCH_SIZE     (1 << LOG_EC_BATCH_SIZE)

/**
 * @class EcAddTask
 * @brief class deriving `TaskBase` to allow use of tasks_manager.h for MSM
 * (The task to be executed is an elliptic curve addition).
 */
template <typename A, typename P>
class EcAddTask : public TaskBase
{
public:
  /**
   * @brief constructor for the task ensuring points are zeros and other fields are init to invalid values (to avoid
   * falsely handling a result at the start). The execution adds points m_point1,m_point2 and stores the result in
   * m_point1.
   */
  EcAddTask()
      : TaskBase(), m_a_points(EC_BATCH_SIZE, P::zero()), m_b_points(EC_BATCH_SIZE, P::zero()),
        m_b_point_ptrs(EC_BATCH_SIZE, nullptr), m_b_affine_points(EC_BATCH_SIZE, A::zero()),
        m_return_idx(EC_BATCH_SIZE, -1), m_opcodes(EC_BATCH_SIZE, ADD_P1_P2_BY_VALUE), m_is_line(EC_BATCH_SIZE, true),
        m_nof_valid_points(0)
  {
  }

  /**
   * @brief Function to be executed by the tasks manager. It can be configured according to the value odf
   */
  void execute()
  {
    static int counter = 0;
    for (int i = 0; i < m_nof_valid_points; i++) {
      switch (m_opcodes[i]) {
      case ADD_P1_P2_BY_VALUE:
        m_a_points[i] = m_a_points[i] + m_b_points[i];
        continue;
      case ADD_P1_AND_P2_POINTER:
        m_a_points[i] = m_a_points[i] + *(m_b_point_ptrs[i]);
        continue;
      case ADD_P1_AND_P2_AFFINE:
        m_a_points[i] = m_a_points[i] + m_b_affine_points[i];
        // continue;
      }
    }
  }

  /**
   * @brief Dispatch task even if batch is not full. This function is mostly used in ends of phases where the left
   * additions are not enough to fill a batch.
   */
  void dispatch_if_not_empty()
  {
    if (is_idle() && m_nof_valid_points > 0) { dispatch(); }
  }

  /**
   * @brief Set up phase 1 addition between a new base and the target bucket.
   * @param bucket - bucket to be added. It isn't passed as Per as the bucket's value can be changed during
   * addition execution.
   * @param base - affine base to be added. Passed as a Per as the input is constant and it saves copying.
   * @param negate_affine - flag to indicate that the base needs to be subbed instead of added.
   * @param is_montgomery - flag to indicate that the base is in Montgomery form and first needs to be converted.
   */
  void set_phase1_addition_with_affine(const P& bucket, const A& base, int bucket_idx)
  {
    m_a_points[m_nof_valid_points] = bucket;
    m_b_affine_points[m_nof_valid_points] = base;
    m_return_idx[m_nof_valid_points] = bucket_idx;
    m_opcodes[m_nof_valid_points] = ADD_P1_AND_P2_AFFINE;

    m_nof_valid_points++;
    if (m_nof_valid_points == EC_BATCH_SIZE) { dispatch(); }
  }

  /**
   * @brief Set up phase 2 addition by value. It is required for the first set of additions - the first line addition
   * of each segment.
   * @param line_sum - line sum to be copied because it is also given to triangle sum.
   * @param bucket - bucket to be added to the line.
   * @param segment_idx - relevant segment to identify the returning result.
   */
  void set_phase2_addition_by_value(const P& line_sum, const P& bucket, int segment_idx)
  {
    m_a_points[m_nof_valid_points] = line_sum;
    m_b_points[m_nof_valid_points] = bucket;
    m_return_idx[m_nof_valid_points] = segment_idx;
    m_is_line[m_nof_valid_points] = true;
    m_opcodes[m_nof_valid_points] = ADD_P1_P2_BY_VALUE;

    m_nof_valid_points++;
    if (m_nof_valid_points == EC_BATCH_SIZE) { dispatch(); }
  }

  /**
   * @brief Set up line addition for phase 2.
   * @param line_sum - line sum to be copied because it is also given to triangle sum.
   * @param bucket_ptr - Per to the bucket to be added to the line, passed as a Per to avoid copying.
   * @param segment_idx - relevant segment to identify the returning result.
   */
  void set_phase2_line_addition(const P& line_sum, P* bucket_ptr, const int segment_idx)
  {
    m_a_points[m_nof_valid_points] = line_sum;
    m_b_point_ptrs[m_nof_valid_points] = bucket_ptr;
    m_return_idx[m_nof_valid_points] = segment_idx;
    m_is_line[m_nof_valid_points] = true;
    m_opcodes[m_nof_valid_points] = ADD_P1_AND_P2_POINTER;

    m_nof_valid_points++;
    if (m_nof_valid_points == EC_BATCH_SIZE) { dispatch(); }
  }

  /**
   * @brief Set up triangle addition for phase 2.
   * No need to specify segment idx as this is only assigned to a task already handling the triangle's segment idx
   * (A previous line or triangle sum of this segment).
   * @param line_sum - line sum to be copied because its value can be updated during the triangle sum execution.
   * @param triangle_sum_ptr - Per to the triangle sum, passed as a Per to avoid unnecessary copying.
   */
  void set_phase2_triangle_addition(P& line_sum, P* triangle_sum_ptr)
  {
    m_a_points[m_nof_valid_points] = line_sum;
    m_b_point_ptrs[m_nof_valid_points] = triangle_sum_ptr;
    m_is_line[m_nof_valid_points] = false;
    m_opcodes[m_nof_valid_points] = ADD_P1_AND_P2_POINTER;

    m_nof_valid_points++;
    if (m_nof_valid_points == EC_BATCH_SIZE) { dispatch(); }
  }

  /**
   * @brief Chain addition used in phase 1 when a collision between a new result and an occupied bucket.
   * @param result - the previous EC addition result.
   * @param bucket - the bucket value to be added to the existing result in p1.
   * @param segment_idx - index of the bucket to return the result to.
   */
  void set_phase1_collision_task(const P& result, const P& bucket, const int segment_idx)
  {
    m_a_points[m_nof_valid_points] = result;
    m_b_points[m_nof_valid_points] = bucket;
    m_return_idx[m_nof_valid_points] = segment_idx;
    m_opcodes[m_nof_valid_points] = ADD_P1_P2_BY_VALUE;

    m_nof_valid_points++;
    if (m_nof_valid_points == EC_BATCH_SIZE) { dispatch(); }
  }

  /**
   * @brief Resets task to idle, resetting the valid points counter to 0.
   */
  void reset_idle()
  {
    m_nof_valid_points = 0;
    set_idle();
  }

  int m_nof_valid_points;

  std::vector<P> m_a_points; // One of the addends that also stores addition results.

  std::vector<int> m_return_idx; // Idx allowing manager to figure out where the result belong to.
  std::vector<bool> m_is_line;   // Indicator for phase 2 sums between line sum and triangle sum.

private:
  // Variations of the second addend which will be used depending on the opcode bellow
  std::vector<P> m_b_points;
  std::vector<A> m_b_affine_points;
  std::vector<P*> m_b_point_ptrs;

  enum eAddType { ADD_P1_P2_BY_VALUE, ADD_P1_AND_P2_POINTER, ADD_P1_AND_P2_AFFINE };
  std::vector<eAddType> m_opcodes;
};

/**
 * @class MSM
 * @brief class for solving multi-scalar-multiplication on the cpu.
 * The class is a template depending on the element relevant to MSM (for instance EC P).
 * NOTE The msm runs only if nof_threads * 4 > nof_bms. The user can guarantee the condition by assigning a proper
 * precompute factor that will decrease the number of BMs required to calculate the MSM.
 */
template <typename A, typename P>
class Msm
{
public:
  /**
   * @brief Constructor for Msm class.
   * @param config - msm config. important parameters that are part of the config extension are: . NOTE: ensure c
   * doesn't divide the scalar width without a remainder.
   * @param c - c separately after cpu_msm handled problematic c values (less than 1 or dividing scalar_t::NBITS
   * without remainder)
   * @param nof_threads - number of worker threads for EC additions.
   */
  Msm(const MSMConfig& config, const int c, const int nof_threads);

  /**
   * @brief Destructor for Msm class.
   * Ensures phase 3 threads have finished and joins them (other deletion are implemented implicitly).
   */
  ~Msm()
  {
    for (std::thread& phase3_thread : m_phase3_threads) {
      phase3_thread.join();
    }
    // thread_f.close();
  }

  /**
   * @brief Main function to execute MSM computation. Launches the 3 phases implemented in the functions below
   * (accumulation, bm sums, final accumulator).
   * @param scalars - Input scalars for MSM.
   * @param bases - EC P input, affine representation.
   * @param msm_size - Size of the above arrays, as they are given as Pers.
   * @param batch_idx - number of current MSM in the batch.
   * @param results - Per to P array in which to store the results. NOTE: the user is expected to preallocate
   * the results array.
   */
  void run_msm(
    const scalar_t* scalars, const A* bases, const unsigned int msm_size, const unsigned int batch_idx, P* results);

private:
  TasksManager<EcAddTask<A, P>> manager; // Tasks manager for multithreading

  const unsigned int m_c;                 // Pipenger constant
  const unsigned int m_num_bkts;          // Number of buckets in each bucket module
  const unsigned int m_precompute_factor; // multiplication of bases already calculated trading memory for performance
  const unsigned int m_num_bms;           // Number of bucket modules (windows in Pipenger's algorithm)
  const bool m_are_scalars_mont;          // Are the input scalars in Montgomery representation
  const bool m_are_points_mont;           //  Are the input points in Montgomery representation
  const int m_batch_size;

  EcAddTask<A, P>* m_curr_task;

  // Phase 1 members
  std::vector<P> m_buckets;           // Vector of all buckets required for phase 1 (All bms in order)
  std::vector<bool> m_bkts_occupancy; // Boolean vector indicating if the corresponding bucket is occupied

  // Phase 2 members
  const int m_log_num_segments;
  const int m_num_bm_segments; // Number of segments in a BM - to maximize parallel operation from the serial bm sums.
  const int m_segment_size;

  /**
   * @struct BmSumSegment
   * @brief Struct bundling the required data members for each BM segment in phase 2.
   */
  struct BmSumSegment {
    // Imagining the required BM sum as a right angle triangle - Nth element is a N high column of the element -
    // A method to summing the BM serially is starting from the top with a triangle sum up to the current line and a
    // line sum which is calculated for each line then added to the triangle.
    // Therefore both sums are need as a part of this struct.
    P triangle_sum;
    P line_sum;

    int m_nof_received_sums = 1; // Counter for numbers of sum received in this row idx - used to determine when a new
                                 // row idx can be dispatched (Due to it depending on the two sums from the prev idx).
    int m_idx_in_segment;        // Counter counting down how far the current sum is through the segment.
    int m_segment_mem_start;     // Offset of segment start in memory / vector.
  };

  // Phase 3 members
  std::vector<std::thread> m_phase3_threads;

  /**
   * @brief Phase 1 (accumulation) of MSM - sorting input points to buckets according to corresponding input scalars.
   * @param scalars - scalar input.
   * @param bases - EC P input, affine representation (Regardless of the defined P type of the class).
   * @param msm_size - Size of the above arrays, as they are given as Pers.
   */
  void phase1_bucket_accumulator(const scalar_t* scalars, const A* bases, const unsigned int msm_size);

  /**
   * @brief Push addition task during phase 1.
   * Th function also handles completed addition results while attempting to insert the new addition task (including
   * potential collision that requires more urgent ec-addition of the result and the current stored value).
   * @param task_bkt_idx - address in m_buckets in which to store the result in the future.
   * @param bkt - the P from m_buckets.
   * @param base - the P from the input bases
   * @param negate_base - flag to signal the task to subtract base instead of adding it.
   */
  void phase1_push_addition(const unsigned int task_bkt_idx, const P bkt, const A base);
  /**
   * @brief Handles the final results of phase 1 (after no other planned additions are required).
   * The function also handles the potential collision similarly to push_addition above.
   */
  void phase1_wait_for_completion();

  /**
   * @brief Phase 2 of MSM. Function handling the initial parallel part of summing the bucket modules. It splits the
   * BMs into segments, summing each separately and passing the segments sum to be handled by the final accumulator.
   * @param segments vector to contain segments line and triangle sums for the final accumulator (output of function).
   */
  void phase2_bm_sum(std::vector<BmSumSegment>& segments);

  /**
   * @brief Setting up phase 2 class members according to phase 1 results.
   * NOTE: This setup function also potentially blocks until the previous MSM's phase 3 finishes (Given that in batched
   * MSM, phase 3 is initiated on a separate thread while the next MSM begins phase 1).
   */
  void phase2_setup(std::vector<BmSumSegment>& segments);

  /**
   * @brief Final accumulation required for MSM calculation. the function will either launch a thread to perform the
   * calculation (`phase3_tread` function) or by the main thread, depending on the position in the batch (Last msm or
   * not).
   * @param segments_ptr - pointer to vector containing the segment calculations of phase 2
   * @param idx_in_batch - idx of the current MSM in the batch.
   * @param result - output, Per to write the MSM result to. Memory for the Per has already been allocated by
   * the user.
   */
  void phase3_final_accumulator(std::vector<BmSumSegment>& segments, int idx_in_batch, P* result);

  /**
   * @brief Phase 3 function to be ran by a separate thread (or main in the end of the run) - i.e. without using the
   * tasks manager. It performs the remaining serial addition in each BM and sums them to one final MSM result.
   * @param result - Per to write the MSM result to. Memory for the Per has already been allocated by the user.
   */
  void phase3_thread(std::vector<BmSumSegment> segments, P* result);

  /**
   * @brief Function for resetting class members between batch runs.
   */
  void batch_run_reset() { std::fill(m_bkts_occupancy.begin(), m_bkts_occupancy.end(), false); }
};

template <typename A, typename P>
Msm<A, P>::Msm(const MSMConfig& config, const int c, const int nof_threads)
    : manager(nof_threads), m_curr_task(nullptr),

      m_c(c), m_num_bkts(1 << (m_c - 1)), m_precompute_factor(config.precompute_factor),
      m_num_bms(((scalar_t::NBITS - 1) / (config.precompute_factor * m_c)) + 1),
      m_are_scalars_mont(config.are_scalars_montgomery_form), m_are_points_mont(config.are_points_montgomery_form),
      m_batch_size(config.batch_size),

      m_buckets(m_num_bms * m_num_bkts), m_bkts_occupancy(m_num_bms * m_num_bkts, false),

      m_log_num_segments(std::max(
        (int)std::floor(
          std::log2((double)(nof_threads * TASKS_PER_THREAD * EC_BATCH_SIZE - 1) / (double)(2 * m_num_bms))),
        0)),
      m_num_bm_segments(std::min((int)(1 << m_log_num_segments), (int)(m_num_bkts))),
      m_segment_size(std::max((int)(m_num_bkts >> m_log_num_segments), 1)),

      m_phase3_threads(m_batch_size - 1)
{
}

template <typename A, typename P>
void Msm<A, P>::run_msm(
  const scalar_t* scalars, const A* bases, const unsigned int msm_size, const unsigned int batch_idx, P* results)
{
#ifdef LOG_UTILIZATION
  Timer tmsm("Total msm time");
#endif
  {
#ifdef LOG_UTILIZATION
    Timer tp1("Phase 1");
#endif
    phase1_bucket_accumulator(scalars, bases, msm_size);
  }
  auto segments = std::vector<BmSumSegment>(m_num_bms * m_num_bm_segments);
  {
#ifdef LOG_UTILIZATION
    Timer tp1("Phase 2");
#endif
    phase2_bm_sum(segments);
  }
  {
#ifdef LOG_UTILIZATION
    Timer tp1("Phase 3");
#endif
    phase3_final_accumulator(segments, batch_idx, results);
    if (batch_idx < m_batch_size - 1) { batch_run_reset(); }
  }
}

template <typename A, typename P>
void Msm<A, P>::phase1_bucket_accumulator(const scalar_t* scalars, const A* bases, const unsigned int msm_size)
{
  const int coeff_bit_mask_no_sign_bit = m_num_bkts - 1;
  const int coeff_bit_mask_with_sign_bit = (1 << m_c) - 1;
  // NUmber of windows / additions per scalar in case num_bms * precompute_factor exceed scalar width
  const int num_bms_before_precompute = ((scalar_t::NBITS - 1) / m_c) + 1; // +1 for ceiling
  int carry = 0;
  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    // Handle required preprocess of scalar
    scalar_t scalar = m_are_scalars_mont ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(scalar_t::NBITS - 1, 1) > 0;
    if (negate_p_and_s) { scalar = scalar_t::neg(scalar); }
    for (int j = 0; j < m_precompute_factor; j++) {
      // Handle required preprocess of base P
      A base =
        m_are_points_mont ? A::from_montgomery(bases[m_precompute_factor * i + j]) : bases[m_precompute_factor * i + j];
      // TODO move to preprocess before precompute to avoid repeating conversions
      if (base == A::zero()) { continue; }
      if (negate_p_and_s) { base = A::neg(base); }
      // TODO move to preprocess before precompute to avoid repeating negations

      for (int k = 0; k < m_num_bms; k++) {
        // Avoid seg fault in case precompute_factor*c exceeds the scalar width by comparing index with num additions
        if (m_num_bms * j + k >= num_bms_before_precompute) { break; }

        uint32_t curr_coeff = scalar.get_scalar_digit(m_num_bms * j + k, m_c) + carry;
        int bkt_idx = 0;
        // For the edge case of curr_coeff = c (limb=c-1, carry=1) use the sign bit mask
        if ((curr_coeff & coeff_bit_mask_with_sign_bit) != 0) {
          // Remove sign to infer the bkt idx.
          carry = curr_coeff > m_num_bkts;
          if (!carry) {
            bkt_idx = m_num_bkts * k + (curr_coeff & coeff_bit_mask_no_sign_bit);
          } else {
            bkt_idx = m_num_bkts * k + ((-curr_coeff) & coeff_bit_mask_no_sign_bit);
          }

          // Check for collision in that bucket and either dispatch an addition or store the P accordingly.
          if (m_bkts_occupancy[bkt_idx]) {
            m_bkts_occupancy[bkt_idx] = false;
            phase1_push_addition(bkt_idx, m_buckets[bkt_idx], carry > 0 ? A::neg(base) : base);
          } else {
            m_bkts_occupancy[bkt_idx] = true;
            m_buckets[bkt_idx] = carry > 0 ? P::neg(P::from_affine(base)) : P::from_affine(base);
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

template <typename A, typename P>
void Msm<A, P>::phase1_push_addition(const unsigned int task_bkt_idx, const P bkt, const A base)
{
  while (m_curr_task == nullptr) {
    // Use the search for an available (idle or completed) task as an opportunity to handle the existing results.
    m_curr_task = manager.get_idle_or_completed_task();
    if (m_curr_task->is_completed()) {
      const int nof_results = m_curr_task->m_nof_valid_points;
      m_curr_task->reset_idle();
      for (int i = 0; i < nof_results; i++) {
        // Check for collision in the destination bucket, and chain and addition / store result accordingly.
        if (m_bkts_occupancy[m_curr_task->m_return_idx[i]]) {
          m_bkts_occupancy[m_curr_task->m_return_idx[i]] = false;
          m_curr_task->set_phase1_collision_task(
            m_curr_task->m_a_points[i], m_buckets[m_curr_task->m_return_idx[i]], m_curr_task->m_return_idx[i]);
        } else {
          m_buckets[m_curr_task->m_return_idx[i]] = m_curr_task->m_a_points[i];
          m_bkts_occupancy[m_curr_task->m_return_idx[i]] = true;
        }
      }
    }
    // If the collision tasks cause the task to be dispatched again this task can't be assigned more additions -
    // repeat the loop with a new task.
    if (!m_curr_task->is_idle()) { m_curr_task = nullptr; }
  }
  // After handling the result a new one can be set.
  m_curr_task->set_phase1_addition_with_affine(bkt, base, task_bkt_idx);
  if (!m_curr_task->is_idle()) { m_curr_task = nullptr; }
}

template <typename A, typename P>
void Msm<A, P>::phase1_wait_for_completion()
{
  // In case remaining additions are smaller than a batch size - dispatch current task
  if (m_curr_task && m_curr_task->is_idle()) { m_curr_task->dispatch(); }

  EcAddTask<A, P>* task = manager.get_completed_task();
  while (task != nullptr) {
    const int nof_results = task->m_nof_valid_points;
    task->reset_idle();
    bool had_collision = false;
    for (int i = 0; i < nof_results; i++) {
      // Check for collision in the destination bucket, and chain and addition / store result accordingly.
      if (m_bkts_occupancy[task->m_return_idx[i]]) {
        m_bkts_occupancy[task->m_return_idx[i]] = false;
        task->set_phase1_collision_task(task->m_a_points[i], m_buckets[task->m_return_idx[i]], task->m_return_idx[i]);
        had_collision = true;
      } else {
        m_buckets[task->m_return_idx[i]] = task->m_a_points[i];
        m_bkts_occupancy[task->m_return_idx[i]] = true;
      }
    }
    task->dispatch_if_not_empty();

    task = manager.get_completed_task();
  }
}

template <typename A, typename P>
void Msm<A, P>::phase2_bm_sum(std::vector<BmSumSegment>& segments)
{
  phase2_setup(segments);
  if (m_segment_size > 1) {
    // Send first additions - line additions.
    for (int i = 0; i < m_num_bms * m_num_bm_segments; i++) {
      if (i % EC_BATCH_SIZE == 0) { m_curr_task = manager.get_idle_task(); }
      BmSumSegment& curr_segment = segments[i]; // For readability

      int bkt_idx = curr_segment.m_segment_mem_start + curr_segment.m_idx_in_segment;
      P bucket = m_bkts_occupancy[bkt_idx] ? m_buckets[bkt_idx] : P::zero();
      m_curr_task->set_phase2_addition_by_value(curr_segment.line_sum, bucket, i);
    }
    // Dispatch last task if itis not enough to fill a batch and dispatch automatically.
    if (m_curr_task->is_idle()) { m_curr_task->dispatch(); }

    // Loop until all line/tri sums are done.
    int done_segments = 0;
    while (done_segments < m_num_bms * m_num_bm_segments) {
      m_curr_task = manager.get_completed_task();

      // Check if there is a need for a line task according to the received sums counter of one of the received segments
      EcAddTask<A, P>* line_task = nullptr;
      if (segments[m_curr_task->m_return_idx[0]].m_nof_received_sums == 1) { line_task = manager.get_idle_task(); }

      const int nof_results = m_curr_task->m_nof_valid_points;
      m_curr_task->reset_idle();

      for (int i = 0; i < nof_results; i++) {
        BmSumSegment& curr_segment = segments[m_curr_task->m_return_idx[i]]; // For readability

        if (m_curr_task->m_is_line[i]) {
          curr_segment.line_sum = m_curr_task->m_a_points[i];
        } else {
          curr_segment.triangle_sum = m_curr_task->m_a_points[i];
        }
        curr_segment.m_nof_received_sums++;

        // Check if this was the last addition in the segment
        if (curr_segment.m_idx_in_segment < 0) {
          done_segments++;
          continue;
        }
        // Otherwise check if it is possible to assign new additions:
        // Triangle sum is dependent on the 2 previous sums (line and triangle) - so check if 2 sums were received.
        // Line sum (if not the last one in the segment).

        // Due to the choice of num segments being less than half of total tasks there ought to be an idle task
        // for the line sum.
        if (curr_segment.m_nof_received_sums == 2) {
          curr_segment.m_nof_received_sums = 0;
          m_curr_task->set_phase2_triangle_addition(curr_segment.line_sum, &(curr_segment.triangle_sum));
          curr_segment.m_idx_in_segment--;

          if (curr_segment.m_idx_in_segment >= 0) {
            int bkt_idx = curr_segment.m_segment_mem_start + curr_segment.m_idx_in_segment;
            if (m_bkts_occupancy[bkt_idx]) {
              int return_idx = m_curr_task->m_return_idx[i];
              line_task->set_phase2_line_addition(curr_segment.line_sum, &m_buckets[bkt_idx], return_idx);
            } else {
              // curr_segment.m_nof_received_sums++;
              int return_idx = m_curr_task->m_return_idx[i];
              line_task->set_phase2_addition_by_value(curr_segment.line_sum, P::zero(), return_idx);
            } // No need to add a zero - just increase nof_received_sums.
          }
        }
      }
      // Check if tri and line task haven't been dispatched due to not enough inputs - dispatch them
      m_curr_task->dispatch_if_not_empty();
      if (line_task) { line_task->dispatch_if_not_empty(); }
    }
  }
}

template <typename A, typename P>
void Msm<A, P>::phase2_setup(std::vector<BmSumSegment>& segments)
{
  // Init values of partial (line) and total (triangle) sums
  for (int i = 0; i < m_num_bms; i++) {
    for (int j = 0; j < m_num_bm_segments - 1; j++) {
      BmSumSegment& segment = segments[m_num_bm_segments * i + j];
      int bkt_idx = m_num_bkts * i + m_segment_size * (j + 1);
      if (m_bkts_occupancy[bkt_idx]) {
        segment.triangle_sum = m_buckets[bkt_idx];
        segment.line_sum = m_buckets[bkt_idx];
      } else {
        segment.triangle_sum = P::zero();
        segment.line_sum = P::zero();
      }
      segment.m_idx_in_segment = m_segment_size - 2;
      segment.m_segment_mem_start = m_num_bkts * i + m_segment_size * j + 1;
    }

    // The most significant bucket of every bm is stored in address 0 -
    // so the last tri/line sums will be initialized to bucket[0]
    BmSumSegment& segment = segments[m_num_bm_segments * (i + 1) - 1];
    int bkt_idx = m_num_bkts * i;
    if (m_bkts_occupancy[bkt_idx]) {
      segment.triangle_sum = m_buckets[bkt_idx];
      segment.line_sum = m_buckets[bkt_idx];
    } else {
      segment.triangle_sum = P::zero();
      segment.line_sum = P::zero();
    }
    segment.m_idx_in_segment = m_segment_size - 2;
    segment.m_segment_mem_start = m_num_bkts * (i + 1) - m_segment_size + 1;
  }
}

template <typename A, typename P>
void Msm<A, P>::phase3_final_accumulator(std::vector<BmSumSegment>& segments, int idx_in_batch, P* result)
{
  // If it isn't the last MSM in the batch - run phase 3 on a separate thread to start utilizing the tasks manager on
  // the next phase 1.
  if (idx_in_batch == m_batch_size - 1) {
    phase3_thread(segments, result);
  } else {
    m_phase3_threads[idx_in_batch] = std::thread(&Msm<A, P>::phase3_thread, this, std::move(segments), result);
  }
}

template <typename A, typename P>
void Msm<A, P>::phase3_thread(std::vector<BmSumSegment> segments, P* result)
{
  for (int i = 0; i < m_num_bms; i++) {
    // Weighted sum of all the lines for each bm - summed in a similar fashion of the triangle sum of phase 2
    P partial_sum = segments[m_num_bm_segments * (i + 1) - 1].line_sum;
    P total_sum = segments[m_num_bm_segments * (i + 1) - 1].line_sum;
    for (int j = m_num_bm_segments - 2; j > 0; j--) {
      partial_sum = partial_sum + segments[m_num_bm_segments * i + j].line_sum;
      total_sum = total_sum + partial_sum;
    }
    segments[m_num_bm_segments * i].line_sum = total_sum;

    // Convert weighted lines sum to rectangles sum by doubling
    int num_doubles = m_c - 1 - m_log_num_segments;
    for (int k = 0; k < num_doubles; k++) {
      segments[m_num_bm_segments * i].line_sum = P::dbl(segments[m_num_bm_segments * i].line_sum);
    }

    // Sum triangles within bm linearly
    for (int j = 1; j < m_num_bm_segments; j++) {
      segments[m_num_bm_segments * i].triangle_sum =
        segments[m_num_bm_segments * i].triangle_sum + segments[m_num_bm_segments * i + j].triangle_sum;
    }

    // After which add the lines and triangle sums to one sum of the entire BM
    if (m_num_bm_segments > 1) {
      segments[m_num_bm_segments * i].triangle_sum =
        segments[m_num_bm_segments * i].triangle_sum + segments[m_num_bm_segments * i].line_sum;
    }
  }

  // Sum BM sums together
  P final_sum = segments[(m_num_bms - 1) * m_num_bm_segments].triangle_sum;
  for (int i = m_num_bms - 2; i >= 0; i--) {
    // Multiply by the BM digit factor 2^c - i.e. c doublings
    for (int j = 0; j < m_c; j++) {
      final_sum = P::dbl(final_sum);
    }
    final_sum = final_sum + segments[m_num_bm_segments * i].triangle_sum;
  }
  *result = final_sum;
}

// None class functions below:

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
  int c = config.c;
  if (c < 1) { c = std::max((int)std::log2(msm_size) - 1, 8); }

  int nof_threads = std::thread::hardware_concurrency() - 1;
  if (config.ext && config.ext->has(CpuBackendConfig::CPU_NOF_THREADS)) {
    nof_threads = config.ext->get<int>(CpuBackendConfig::CPU_NOF_THREADS);
  }
  if (nof_threads <= 0) {
    ICICLE_LOG_WARNING << "Unable to detect number of hardware supported threads - fixing it to 1\n";
    nof_threads = 1;
  }
  auto msm = Msm<A, P>{config, c, nof_threads};

  for (int i = 0; i < config.batch_size; i++) {
    int batch_start_idx = msm_size * i;
    int bases_start_idx = config.are_points_shared_in_batch ? 0 : batch_start_idx;
    msm.run_msm(&scalars[batch_start_idx], &bases[bases_start_idx], msm_size, i, &results[i]);
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
  int c = config.c;
  if (c < 1) { c = std::max((int)std::log2(nof_bases) - 1, 8); }

  int precompute_factor = config.precompute_factor;
  bool is_mont = config.are_points_montgomery_form;
  const unsigned int num_bms_no_precomp = (scalar_t::NBITS - 1) / c + 1;
  const unsigned int shift = c * ((num_bms_no_precomp - 1) / precompute_factor + 1);
  for (int i = 0; i < nof_bases; i++) {
    output_bases[precompute_factor * i] = input_bases[i];
    P point = P::from_affine(is_mont ? A::from_montgomery(input_bases[i]) : input_bases[i]);
    for (int j = 1; j < precompute_factor; j++) {
      for (int k = 0; k < shift; k++) {
        point = P::dbl(point);
      }
      output_bases[precompute_factor * i + j] = is_mont ? A::to_montgomery(P::to_affine(point)) : P::to_affine(point);
    }
  }
  return eIcicleError::SUCCESS;
}