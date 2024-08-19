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
template <typename Point>
class EcAddTask : public TaskBase
{
public:
  /**
   * @brief constructor for the task ensuring points are zeros and other fields are init to invalid values (to avoid
   * falsely handling a result at the start). The execution adds points m_p1,m_p2 and stores the result in m_p1.
   */
  EcAddTask() : TaskBase(), m_p1(Point::zero()), m_p2(Point::zero()), m_return_idx(-1) {}

  /**
   * @brief Function to be executed by the tasks manager. It can be configured according to the value odf
   */
  void execute()
  {
    switch (m_p2_config) {
    case ADD_P1_P2_BY_VALUE:
      m_p1 = m_p1 + m_p2;
      return;
    case ADD_P1_AND_P2_POINTER:
      m_p1 = m_p1 + *m_p2_pointer;
      return;
    case ADD_P1_AND_AFFINE_P1:
      m_p1 = m_p1 + m_p2_affine;
      return;
    }
  }

  /**
   * @brief Set up phase 1 addition between a new base and the target bucket.
   * @param bucket - bucket to be added. It isn't passed as pointer as the bucket's value can be changed during
   * addition execution.
   * @param base - affine base to be added. Passed as a pointer as the input is constant and it saves copying.
   * @param negate_affine - flag to indicate that the base needs to be subbed instead of added.
   * @param is_montgomery - flag to indicate that the base is in Montgomery form and first needs to be converted.
   */
  void set_phase1_addition_with_affine(const Point& bucket, const affine_t base, int bucket_idx)
  {
    m_p1 = bucket;
    m_p2_affine = base;
    m_return_idx = bucket_idx;
    m_p2_config = ADD_P1_AND_AFFINE_P1;
    dispatch();
  }

  /**
   * @brief Set up phase 2 addition by value. It is required for the first set of additions - the first line addition
   * of each segment.
   * @param line_sum - line sum to be copied because it is also given to triangle sum.
   * @param bucket - bucket to be added to the line.
   * @param segment_idx - relevant segment to identify the returning result.
   */
  void set_phase2_addition_by_value(const Point& line_sum, const Point& bucket, int segment_idx)
  {
    m_p1 = line_sum;
    m_p2 = bucket;
    m_return_idx = segment_idx;
    m_is_line = true;
    m_p2_config = ADD_P1_P2_BY_VALUE;
    dispatch();
  }

  /**
   * @brief Set up line addition for phase 2.
   * @param line_sum - line sum to be copied because it is also given to triangle sum.
   * @param bucket_ptr - pointer to the bucket to be added to the line, passed as a pointer to avoid copying.
   * @param segment_idx - relevant segment to identify the returning result.
   */
  void set_phase2_line_addition(const Point& line_sum, Point* bucket_ptr, const int segment_idx)
  {
    m_p1 = line_sum;
    m_p2_pointer = bucket_ptr;
    m_return_idx = segment_idx;
    m_is_line = true;
    m_p2_config = ADD_P1_AND_P2_POINTER;
    dispatch();
  }

  /**
   * @brief Set up triangle addition for phase 2.
   * No need to specify segment idx as this is only assigned to a task already handling the triangle's segment idx
   * (A previous line or triangle sum of this segment).
   * @param line_sum - line sum to be copied because its value can be updated during the triangle sum execution.
   * @param triangle_sum_ptr - pointer to the triangle sum, passed as a pointer to avoid unnecessary copying.
   */
  void set_phase2_triangle_addition(Point& line_sum, Point* triangle_sum_ptr)
  {
    m_p1 = line_sum;
    m_p2_pointer = triangle_sum_ptr;
    m_is_line = false;
    m_p2_config = ADD_P1_AND_P2_POINTER;
    dispatch();
  }

  /**
   * @brief Chain addition used in phase 1 when a collision between a new result and an occupied bucket.
   * @param bucket - the bucket value to be added to the existing result in p1.
   */
  void set_phase1_collision_task(Point& bucket)
  {
    m_p2 = bucket;
    m_p2_config = ADD_P1_P2_BY_VALUE;
    dispatch();
  }

  Point m_p1;             // One of the addends, and holds the addition result afterwards
  int m_return_idx;       // Idx allowing manager to figure out where the result belong to.
  bool m_is_line = false; // Indicator for phase 2 sums between line sum and triangle sum.

private:
  enum eAddType { ADD_P1_P2_BY_VALUE, ADD_P1_AND_P2_POINTER, ADD_P1_AND_AFFINE_P1 };
  eAddType m_p2_config = ADD_P1_P2_BY_VALUE;

  // Various configs of the second addend p2, one for each eAddType
  Point m_p2;
  Point const* m_p2_pointer;
  affine_t m_p2_affine;
};

/**
 * @class MSM
 * @brief class for solving multi-scalar-multiplication on the cpu.
 * The class is a template depending on the element relevant to MSM (for instance EC point).
 * NOTE The msm runs only if nof_threads * 4 > nof_bms. The user can guarantee the condition by assigning a proper
 * precompute factor that will decrease the number of BMs required to calculate the MSM.
 */
template <typename Point>
class Msm
{
public:
  /**
   * @brief Constructor for Msm class.
   * @param config - msm config. important parameters that are part of the config extension are: n_threads, c
   */
  Msm(const MSMConfig& config);

  /**
   * @brief Destructor for Msm class.
   * Ensures phase 3 threads have finished and joins them (other deletion are implemented implicitly).
   */
  ~Msm()
  {
    for (std::thread& p3_thread : m_p3_threads) {
      p3_thread.join();
    }
  }

  /**
   * @brief Main function to execute MSM computation.
   * @param scalars - Input scalars for MSM.
   * @param bases - EC point input, affine representation.
   * @param msm_size - Size of the above arrays, as they are given as pointers.
   * @param batch_idx - number of current MSM in the batch.
   * @param results - pointer to Point array in which to store the results. NOTE: the user is expected to preallocate
   * the results array.
   */
  void run_msm(
    const scalar_t* scalars,
    const affine_t* bases,
    const unsigned int msm_size,
    const unsigned int batch_idx,
    Point* results);

private:
  TasksManager<EcAddTask<Point>> manager; // Tasks manager for multithreading

  const unsigned int m_c;                 // Pipenger constant
  const unsigned int m_num_bkts;          // Number of buckets in each bucket module
  const unsigned int m_precompute_factor; // multiplication of points already calculated trading memory for performance
  const unsigned int m_num_bms;           // Number of bucket modules (windows in Pipenger's algorithm)
  const bool m_are_scalars_mont;          // Are the input scalars in Montgomery representation
  const bool m_are_points_mont;           //  Are the input points in Montgomery representation
  const int m_batch_size;

  // Phase 1 members
  std::vector<Point> m_buckets;       // Vector of all buckets required for phase 1 (All bms in order)
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
    Point triangle_sum;
    Point line_sum;

    int m_nof_received_sums = 1; // Counter for numbers of sum received in this row idx - used to determine when a new
                                 // row idx can be dispatched (Due to it depending on the two sums from the prev idx).
    int m_idx_in_segment;        // Counter counting down how far the current sum is through the segment.
    int m_segment_mem_start;     // Offset of segment start in memory / vector.
  };

  // Phase 3 members
  std::vector<std::thread> m_p3_threads;

  /**
   * @brief Phase 1 (accumulation) of MSM
   * @param scalars - scalar input.
   * @param bases - EC point input, affine representation (Regardless of the defined Point type of the class).
   * @param msm_size - Size of the above arrays, as they are given as pointers.
   */
  void bucket_accumulator(const scalar_t* scalars, const affine_t* bases, const unsigned int msm_size);

  /**
   * @brief Push addition task during phase 1.
   * Th function also handles completed addition results while attempting to insert the new addition task (including
   * potential collision that requires more urgent ec-addition of the result and the current stored value).
   * @param task_bkt_idx - address in m_buckets in which to store the result in the future.
   * @param bkt - the point from m_buckets.
   * @param base - the point from the input bases
   * @param negate_base - flag to signal the task to subtract base instead of adding it.
   */
  void phase1_push_addition(const unsigned int task_bkt_idx, const Point bkt, const affine_t base);
  /**
   * @brief Handles the final results of phase 1 (after no other planned additions are required).
   * The function also handles the potential collision similarly to push_addition above.
   */
  void phase1_wait_for_completion();

  /**
   * @brief Phase 2 of MSM. Function handling the initial parallel part of summing the bucket modules. It splits the
   * BMs into segments, summing each separately and passing the segments sum to be handled by the final accumulator.
   * @return vector containing segments line and triangle sums for the final accumulator.
   */
  void bm_sum(std::shared_ptr<std::vector<BmSumSegment>>& segments_ptr);

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
   * @param result - output, pointer to write the MSM result to. Memory for the pointer has already been allocated by
   * the user.
   */
  void final_accumulator(std::shared_ptr<std::vector<BmSumSegment>>& segments_ptr, int idx_in_batch, Point* result);

  /**
   * @brief Phase 3 function to be ran by a separate thread (or main in the end of the run) - i.e. without using the
   * tasks manager. It performs the remaining serial addition in each BM and sums them to one final MSM result.
   * @param result - pointer to write the MSM result to. Memory for the pointer has already been allocated by the user.
   */
  void phase3_thread(std::shared_ptr<std::vector<BmSumSegment>> segments_ptr, Point* result);

  /**
   * @brief Function for resetting class members between batch runs.
   */
  void batch_run_reset() { std::fill(m_bkts_occupancy.begin(), m_bkts_occupancy.end(), false); }
};

template <typename Point>
Msm<Point>::Msm(const MSMConfig& config)
    : manager(
        config.ext->get<int>("n_threads") > 0 ? config.ext->get<int>("n_threads")
                                              : std::thread::hardware_concurrency()),

      m_c(config.ext->get<int>("c")), m_num_bkts(1 << (m_c - 1)), m_precompute_factor(config.precompute_factor),
      m_num_bms(((scalar_t::NBITS - 1) / (config.precompute_factor * m_c)) + 1),
      m_are_scalars_mont(config.are_scalars_montgomery_form), m_are_points_mont(config.are_points_montgomery_form),
      m_batch_size(config.batch_size),

      m_buckets(m_num_bms * m_num_bkts), m_bkts_occupancy(m_num_bms * m_num_bkts, false),

      m_log_num_segments(std::max(
        (int)std::floor(
          std::log2((double)(config.ext->get<int>("n_threads") * TASKS_PER_THREAD - 1) / (double)(2 * m_num_bms))),
        0)),
      m_num_bm_segments(std::min((int)(1 << m_log_num_segments), (int)(m_num_bms * m_num_bkts))),
      m_segment_size(std::max((int)(m_num_bkts >> m_log_num_segments), 1)),

      m_p3_threads(m_batch_size - 1)
{
}

template <typename Point>
void Msm<Point>::run_msm(
  const scalar_t* scalars,
  const affine_t* bases,
  const unsigned int msm_size,
  const unsigned int batch_idx,
  Point* results)
{
  bucket_accumulator(scalars, bases, msm_size);
  auto segments = std::make_shared<std::vector<BmSumSegment>>(m_num_bms * m_num_bm_segments);
  bm_sum(segments);
  final_accumulator(segments, batch_idx, results);
  if (batch_idx < m_batch_size - 1) { batch_run_reset(); }
}

template <typename Point>
void Msm<Point>::bucket_accumulator(const scalar_t* scalars, const affine_t* bases, const unsigned int msm_size)
{
  const int coeff_bit_mask_no_sign_bit = m_num_bkts - 1;
  const int coeff_bit_mask_with_sign_bit = (1 << m_c) - 1;
  // NUmber of windows / additions per scalar in case num_bms * precompute_factor exceed scalar width
  const int num_additions_per_scalar = (scalar_t::NBITS - 1) / m_c; // +1 for ceiling than -1 for m1

  int carry = 0;
  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    // Handle required preprocess of scalar
    scalar_t scalar = m_are_scalars_mont ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(scalar_t::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = scalar_t::neg(scalar);
    for (int j = 0; j < m_precompute_factor; j++) {
      // Handle required preprocess of base point
      affine_t base = m_are_points_mont ? affine_t::from_montgomery(bases[m_precompute_factor * i + j])
                                        : bases[m_precompute_factor * i + j];
      if (negate_p_and_s) { base = affine_t::neg(base); }

      for (int k = 0; k < m_num_bms; k++) {
        // Avoid seg fault in case precompute_factor*c exceeds the scalar width by comparing index with num additions
        if (m_num_bms * j + k > num_additions_per_scalar) { break; }

        uint32_t curr_coeff = scalar.get_scalar_digit(m_num_bms * j + k, m_c) + carry;
        int bkt_idx = 0;
        if ((curr_coeff & coeff_bit_mask_with_sign_bit) != 0) {
          // Remove sign to infer the bkt idx.
          carry = curr_coeff >= m_num_bkts;
          if (curr_coeff < m_num_bkts) {
            bkt_idx = m_num_bkts * k + curr_coeff;
          } else {
            bkt_idx = m_num_bkts * k + ((-curr_coeff) & coeff_bit_mask_no_sign_bit);
          }

          // Check for collision in that bucket and either dispatch an addition or store the point accordingly.
          if (m_bkts_occupancy[bkt_idx]) {
            m_bkts_occupancy[bkt_idx] = false;
            phase1_push_addition(bkt_idx, m_buckets[bkt_idx], carry > 0 ? affine_t::neg(base) : base);
          } else {
            affine_t base = m_are_points_mont ? affine_t::from_montgomery(bases[m_precompute_factor * i + j])
                                              : bases[m_precompute_factor * i + j];
            if (negate_p_and_s) { base = affine_t::neg(base); }
            m_bkts_occupancy[bkt_idx] = true;
            m_buckets[bkt_idx] = carry > 0 ? Point::neg(Point::from_affine(base)) : Point::from_affine(base);
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
void Msm<Point>::phase1_push_addition(const unsigned int task_bkt_idx, const Point bkt, const affine_t base)
{
  EcAddTask<Point>* task = nullptr;
  while (task == nullptr) {
    // Use the search for an available (idle or completed) task as an opportunity to handle the existing results.
    task = manager.get_idle_or_completed_task();
    if (task->is_completed()) {
      // Check for collision in the destination bucket, and chain and addition / store result accordingly.
      if (m_bkts_occupancy[task->m_return_idx]) {
        m_bkts_occupancy[task->m_return_idx] = false;
        task->set_phase1_collision_task(m_buckets[task->m_return_idx]);
        task = nullptr;
        continue;
      } else {
        m_buckets[task->m_return_idx] = task->m_p1;
        m_bkts_occupancy[task->m_return_idx] = true;
      }
    }
    // After handling the result a new one can be set.
    task->set_phase1_addition_with_affine(bkt, base, task_bkt_idx);
  }
}

template <typename Point>
void Msm<Point>::phase1_wait_for_completion()
{
  EcAddTask<Point>* task = manager.get_completed_task();
  while (task != nullptr) {
    // Check for collision in the destination bucket, and chain and addition / store result accordingly.
    if (m_bkts_occupancy[task->m_return_idx]) {
      m_bkts_occupancy[task->m_return_idx] = false;
      task->set_phase1_collision_task(m_buckets[task->m_return_idx]);
    } else {
      m_buckets[task->m_return_idx] = task->m_p1;
      m_bkts_occupancy[task->m_return_idx] = true;
      task->set_idle();
    }
    task = manager.get_completed_task();
  }
}

template <typename Point>
void Msm<Point>::bm_sum(std::shared_ptr<std::vector<BmSumSegment>>& segments_ptr)
{
  auto& segments = *segments_ptr; // For readability
  phase2_setup(segments);
  if (m_segment_size > 1) {
    // Send first additions - line additions.
    for (int i = 0; i < m_num_bms * m_num_bm_segments; i++) {
      EcAddTask<Point>* task = manager.get_idle_task();
      BmSumSegment& curr_segment = segments[i]; // For readability

      int bkt_idx = curr_segment.m_segment_mem_start + curr_segment.m_idx_in_segment;
      Point bucket = m_bkts_occupancy[bkt_idx] ? m_buckets[bkt_idx] : Point::zero();
      task->set_phase2_addition_by_value(curr_segment.line_sum, bucket, i);
    }

    // Loop until all line/tri sums are done.
    int done_segments = 0;
    while (done_segments < m_num_bms * m_num_bm_segments) {
      EcAddTask<Point>* task = manager.get_completed_task();
      BmSumSegment& curr_segment = segments[task->m_return_idx]; // For readability

      if (task->m_is_line) {
        curr_segment.line_sum = task->m_p1;
      } else {
        curr_segment.triangle_sum = task->m_p1;
      }
      curr_segment.m_nof_received_sums++;

      // Check if this was the last addition in the segment
      if (curr_segment.m_idx_in_segment < 0) {
        done_segments++;
        task->set_idle();
        continue;
      }
      // Otherwise check if it is possible to assign new additions:
      // Triangle sum is dependent on the 2 previous sums (line and triangle) - so check if 2 sums were received.
      if (curr_segment.m_nof_received_sums == 2) {
        curr_segment.m_nof_received_sums = 0;
        task->set_phase2_triangle_addition(curr_segment.line_sum, &(curr_segment.triangle_sum));
        curr_segment.m_idx_in_segment--;

        // Line sum (if not the last one in the segment)
        if (curr_segment.m_idx_in_segment >= 0) {
          int bkt_idx = curr_segment.m_segment_mem_start + curr_segment.m_idx_in_segment;
          if (m_bkts_occupancy[bkt_idx]) {
            int return_idx = task->m_return_idx;
            // Due to the choice of num segments being less than half of total tasks there ought to be an idle task for
            // the line sum
            task = manager.get_idle_task();
            task->set_phase2_line_addition(curr_segment.line_sum, &m_buckets[bkt_idx], return_idx);
          } else {
            curr_segment.m_nof_received_sums++;
          } // No need to add a zero - just increase nof_received_sums
        }
      } else {
        task->set_idle();
      } // Handling Completed task without dispatching a new one
    }
  }
}

template <typename Point>
void Msm<Point>::phase2_setup(std::vector<BmSumSegment>& segments)
{
  // Init values of partial (line) and total (triangle) sum
  for (int i = 0; i < m_num_bms; i++) {
    for (int j = 0; j < m_num_bm_segments - 1; j++) {
      BmSumSegment& segment = segments[m_num_bm_segments * i + j];
      int bkt_idx = m_num_bkts * i + m_segment_size * (j + 1);
      if (m_bkts_occupancy[bkt_idx]) {
        segment.triangle_sum = m_buckets[bkt_idx];
        segment.line_sum = m_buckets[bkt_idx];
      } else {
        segment.triangle_sum = Point::zero();
        segment.line_sum = Point::zero();
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
      segment.triangle_sum = Point::zero();
      segment.line_sum = Point::zero();
    }
    segment.m_idx_in_segment = m_segment_size - 2;
    segment.m_segment_mem_start = m_num_bkts * (i + 1) - m_segment_size + 1;
  }
}

template <typename Point>
void Msm<Point>::final_accumulator(
  std::shared_ptr<std::vector<BmSumSegment>>& segments_ptr, int idx_in_batch, Point* result)
{
  // If it isn't the last MSM in the batch - run phase 3 on a separate thread to start utilizing the tasks manager on
  // the next phase 1.
  if (idx_in_batch == m_batch_size - 1) {
    phase3_thread(segments_ptr, result);
  } else {
    m_p3_threads[idx_in_batch] = std::thread(&Msm<Point>::phase3_thread, this, segments_ptr, result);
  }
}

template <typename Point>
void Msm<Point>::phase3_thread(std::shared_ptr<std::vector<BmSumSegment>> segments_ptr, Point* result)
{
  auto& segments = *segments_ptr; // For readability
  for (int i = 0; i < m_num_bms; i++) {
    // Weighted sum of all the lines for each bm - summed in a similar fashion of the triangle sum of phase 2
    Point partial_sum = segments[m_num_bm_segments * (i + 1) - 1].line_sum;
    Point total_sum = segments[m_num_bm_segments * (i + 1) - 1].line_sum;
    for (int j = m_num_bm_segments - 2; j > 0; j--) {
      partial_sum = partial_sum + segments[m_num_bm_segments * i + j].line_sum;
      total_sum = total_sum + partial_sum;
    }
    segments[m_num_bm_segments * i].line_sum = total_sum;

    // Convert weighted lines sum to rectangles sum by doubling
    int num_doubles = m_c - 1 - m_log_num_segments;
    for (int k = 0; k < num_doubles; k++) {
      segments[m_num_bm_segments * i].line_sum = Point::dbl(segments[m_num_bm_segments * i].line_sum);
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
  Point final_sum = segments[(m_num_bms - 1) * m_num_bm_segments].triangle_sum;
  for (int i = m_num_bms - 2; i >= 0; i--) {
    // Multiply by the BM digit factor 2^c - i.e. c doublings
    for (int j = 0; j < m_c; j++) {
      final_sum = Point::dbl(final_sum);
    }
    final_sum = final_sum + segments[m_num_bm_segments * i].triangle_sum;
  }
  *result = final_sum;
}

// None class functions below:

/**
 * @brief Function to check the MSM config is valid for calculating in CPU.
 * @return - status if the config is supported or not.
 */
eIcicleError not_supported(const MSMConfig& conf)
{
  if (conf.is_async) { return eIcicleError::INVALID_DEVICE; }
  // There is only host (CPU) therefore the following configs are not supported:
  if (conf.are_scalars_on_device | conf.are_points_on_device | conf.are_results_on_device) {
    return eIcicleError::INVALID_DEVICE;
  }
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

  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext->get<int>("c");
  const unsigned int precompute_factor = config.precompute_factor;
  const int num_bms = ((scalar_t::NBITS - 1) / (precompute_factor * c)) + 1;

  if (config.ext->get<int>("n_threads") <= 0) { return eIcicleError::INVALID_ARGUMENT; }

  for (int i = 0; i < config.batch_size; i++) {
    msm->run_msm(&scalars[msm_size * i], bases, msm_size, i, &results[i]);
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
      output_bases[precompute_factor * i + j] =
        is_mont ? A::to_montgomery(projective_t::to_affine(point)) : projective_t::to_affine(point);
    }
  }
  return eIcicleError::SUCCESS;
}