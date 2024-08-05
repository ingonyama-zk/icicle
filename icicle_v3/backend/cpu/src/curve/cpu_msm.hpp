#pragma once

#include <atomic>
#include <mutex>
#include <tuple>
#include <thread>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h> // TODO remove

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/config_extension.h"
#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"
#include "icicle/msm.h"
#include "tasks_manager.h"

using namespace icicle;
#ifdef DUMMY_TYPES
  using affine_t = DummyPoint;
  using projective_t = DummyPoint;
  using scalar_t = DummyScalar;
#else
  using affine_t = curve_config::affine_t;
  using projective_t = curve_config::projective_t;
  using scalar_t = curve_config::scalar_t;
#endif


// TODO remove / revise when finished testing
class Timer
{
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
  std::string fname;

public:
  Timer(std::string func_name)
  {
    start_point = std::chrono::high_resolution_clock::now();
    fname = func_name;
  }

  ~Timer() { Stop(); }

  void Stop()
  {
    auto end_point = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::time_point_cast<std::chrono::microseconds>(start_point).time_since_epoch().count();
    auto end_time = std::chrono::time_point_cast<std::chrono::microseconds>(end_point).time_since_epoch().count();
    auto duration = end_time - start_time;

    double dur_s = duration * 0.001;
    std::cout << "Time of " << fname << ":\t" << dur_s << "ms\n";
  }
};

template<typename Point, typename AddedPoint>
class ECaddTask : public TaskBase
{
public:
  ECaddTask() : TaskBase(), p1(Point::zero()), p2(AddedPoint::zero()), result(Point::zero()), return_idx(-1) {}
  virtual void execute() { result = p1 + p2; }
  
  Point p1, result; // TODO result will be stored in p1 and support two point types
  AddedPoint p2;
  int return_idx;
};

template <typename Point>
class Msm
{
private:
  // std::vector<WorkThread<Point>> threads;
  TasksManager<ECaddTask<Point, Point>> manager;
  const unsigned int n_threads;

  const unsigned int c;
  const unsigned int num_bkts;
  const unsigned int num_bms;
  const unsigned int precomp_f;
  const bool are_scalars_mont;
  const bool are_points_mont;

  int loop_count = 0;
  int num_additions = 0;

  // Phase 1
  Point* bkts;
  bool* bkts_occupancy;
  // Phase 2
  const int log_num_segments;
  const int num_bm_segments;
  const int segment_size;
  Point* phase2_sums;
  std::tuple<int, int>* task_assigned_to_sum;
  Point* bm_sums;
  // Phase 3
  bool mid_phase3;
  int num_valid_results;
  Point* results;

  std::ofstream bkts_f; // TODO remove files
  std::ofstream trace_f;

  void wait_for_idle();
  // void old_wait_for_idle();

  // template <typename Base>
  // void push_addition( const unsigned int task_bkt_idx,
  //                     const Point bkt,
  //                     const Base& base,
  //                     int pidx,
  //                     Point* result_arr,
  //                     bool* );

  void phase1_push_addition(const unsigned int task_bkt_idx, const Point bkt, const Point& base, int pidx);

  template <typename Base>
  // void old_phase1_push_addition(const unsigned int task_bkt_idx, const Point bkt, const Base& base, int pidx);

  std::tuple<int, int> phase2_push_addition(const unsigned int task_bkt_idx, const Point& bkt, const Point& base);

  void bkt_file(); // TODO remove

public:
  Msm(const MSMConfig& config);
  ~Msm();

  Point* bucket_accumulator(
    const scalar_t* scalars,
    const affine_t* bases,
    const unsigned int msm_size); // TODO change type in the end to void

  Point* bm_sum();
};

// TODO ask for help about memory management before / at C.R.
template <typename Point>
Point* msm_bucket_accumulator(
  const scalar_t* scalars,
  const affine_t* bases,
  const unsigned int c,
  const unsigned int num_bms,
  const unsigned int msm_size,
  const unsigned int precomp_f,
  const bool is_s_mont,
  const bool is_b_mont)
{
  /**
   * Accumulate into the different bkts
   * @param scalars - original scalars given from the msm result
   * @param bases - point bases to add
   * @param c - address width of bucket modules to split scalars above
   * @param msm_size - number of scalars to add
   * @param is_s_mont - flag indicating input scalars are in Montgomery form
   * @param is_b_mont - flag indicating input bases are in Montgomery form
   * @return bkts - points array containing all bkts
   */
  auto t = Timer("P1:bucket-accumulator");
  uint32_t num_bkts = 1 << (c - 1);
  Point* bkts;
  {
    auto t2 = Timer("P1:memory_allocation");
    bkts = new Point[num_bms * num_bkts];
  }
  {
    auto t3 = Timer("P1:memory_init");
    std::fill_n(bkts, num_bkts * num_bms, Point::zero());
  }
  uint32_t coeff_bit_mask = num_bkts - 1;
  const int num_windows_m1 = (scalar_t::NBITS - 1) / c;
  int carry;

#ifdef DEBUG_PRINTS
  std::string trace_fname = "trace_bucket_single.txt";
  std::ofstream trace_f(trace_fname);
  if (!trace_f.good()) {
    std::cout << "ERROR: can't open file:\t" << trace_fname << std::endl;
    return nullptr;
  } // TODO remove log
#endif

  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    scalar_t scalar = is_s_mont ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(scalar_t::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = scalar_t::neg(scalar);
    for (int j = 0; j < precomp_f; j++) {
      affine_t point = is_b_mont ? affine_t::from_montgomery(bases[precomp_f * i + j]) : bases[precomp_f * i + j];
      if (negate_p_and_s) point = affine_t::neg(point);
      for (int k = 0; k < num_bms; k++) {
        // In case precomp_f*c exceeds the scalar width
        if (num_bms * j + k > num_windows_m1) { break; }

        uint32_t curr_coeff = scalar.get_scalar_digit(num_bms * j + k, c) + carry;
        if ((curr_coeff & ((1 << c) - 1)) != 0) {
          if (curr_coeff < num_bkts) {
            int bkt_idx = num_bkts * k + curr_coeff;
#ifdef DEBUG_PRINTS
            if (Point::is_zero(bkts[bkt_idx])) {
              trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << point.x << '\n';
            } else {
              trace_f << '#' << bkt_idx << ":\tRead for addition:\t" << Point::to_affine(bkts[bkt_idx]).x
                      << "\t(With new point:\t" << point.x << " = " << Point::to_affine(bkts[bkt_idx] + point).x
                      << ")\n";
              trace_f << '#' << bkt_idx << ":\tWrite (res) free cell:\t" << Point::to_affine(bkts[bkt_idx] + point).x
                      << '\n';
            } // TODO remove double addition
#endif

            bkts[num_bkts * k + curr_coeff] =
              Point::is_zero(bkts[num_bkts * k + curr_coeff])
                ? Point::from_affine(point)
                : bkts[num_bkts * k + curr_coeff] + point; // TODO change here order of precomp
            carry = 0;
          } else {
            int bkt_idx = num_bkts * k + ((-curr_coeff) & coeff_bit_mask);
#ifdef DEBUG_PRINTS
            if (Point::is_zero(bkts[bkt_idx])) {
              trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << affine_t::neg(point).x << '\n';
            } else {
              trace_f << '#' << bkt_idx << ":\tRead for subtraction:\t" << Point::to_affine(bkts[bkt_idx]).x
                      << "\t(With new point:\t" << point.x << " = " << Point::to_affine(bkts[bkt_idx] - point).x
                      << ")\n";
              trace_f << '#' << bkt_idx << ":\tWrite (res) free cell:\t" << Point::to_affine(bkts[bkt_idx] - point).x
                      << '\n';
            } // TODO remove double addition
#endif

            bkts[num_bkts * k + ((-curr_coeff) & coeff_bit_mask)] =
              Point::is_zero(bkts[num_bkts * k + ((-curr_coeff) & coeff_bit_mask)])
                ? Point::neg(Point::from_affine(point))
                : bkts[num_bkts * k + ((-curr_coeff) & coeff_bit_mask)] - point;
            carry = 1;
          }
        } else {
          carry = curr_coeff >> c; // Edge case for coeff = 1 << c
        }
      }
    }
  }
#ifdef DEBUG_PRINTS
  trace_f.close();
  std::string b_fname = "buckets_single.txt";
  std::ofstream bkts_f_single(b_fname);
  if (!bkts_f_single.good()) {
    std::cout << "ERROR: can't open file:\t" << b_fname << std::endl;
    return nullptr;
  } // TODO remove log
  for (int i = 0; i < num_bms; i++)
    for (int j = 0; j < num_bkts; j++)
      bkts_f_single << '(' << i << ',' << j << "):\t" << Point::to_affine(bkts[num_bkts * i + j]).x << '\n';
  bkts_f_single.close();
#endif
  return bkts;
}

template <typename Point>
Point* msm_bm_sum(Point* bkts, const unsigned int c, const unsigned int num_bms)
{
  /**
   * Sum bucket modules to one point each
   * @param bkts - point array containing all bkts ordered by bucket module
   * @param c - bucket width
   * @param num_bms - number of bucket modules
   * @return bm_sums - point array containing the bucket modules' sums
   */
  auto t = Timer("P2:bucket-module-sum");
  uint32_t num_bkts = 1 << (c - 1); // NOTE implicitly assuming that c<32

  Point* bm_sums = new Point[num_bms];

  for (int k = 0; k < num_bms; k++) {
    bm_sums[k] = Point::copy(bkts[num_bkts * k]); // Start with bucket zero that holds the weight <num_bkts>
    Point partial_sum = Point::copy(bkts[num_bkts * k]);

    for (int i = num_bkts - 1; i > 0; i--) {
      if (!Point::is_zero(bkts[num_bkts * k + i])) partial_sum = partial_sum + bkts[num_bkts * k + i];
      if (!Point::is_zero(partial_sum)) bm_sums[k] = bm_sums[k] + partial_sum;
    }
  }

  return bm_sums;
}

template <typename Point>
Point msm_final_sum(Point* bm_sums, const unsigned int c, const unsigned int num_bms, const bool is_b_mont)
{
  /**
   * Sum the bucket module sums to the final result
   * @param bm_sums - point array containing bucket module sums
   * @param c - bucket module width / shift between subsequent bkts
   * @param is_b_mont - flag indicating input bases are in Montgomery form
   * @return result - msm calculation
   */
  auto t = Timer("P3:final-accumulator");
  Point result = bm_sums[num_bms - 1];
  for (int k = num_bms - 2; k >= 0; k--) {
    if (Point::is_zero(result)) {
      if (!Point::is_zero(bm_sums[k])) result = Point::copy(bm_sums[k]);
    } else {
      for (int dbl = 0; dbl < c; dbl++) {
        result = Point::dbl(result);
      }
      if (!Point::is_zero(bm_sums[k])) result = result + bm_sums[k];
    }
  }
  // auto result_converted = is_b_mont? Point::to_montgomery(result) : result;
  return result;
}

template <typename Point>
void msm_delete_arrays(Point* bkts, Point* bms, const unsigned int num_bms)
{
  // TODO memory management
  delete[] bkts;
  delete[] bms;
}

eIcicleError not_supported(const MSMConfig& c)
{
  /**
   * Check config for tests that are currently not supported
   */
  if (c.batch_size > 1) return eIcicleError::INVALID_ARGUMENT; // TODO add support
  if (c.are_scalars_on_device | c.are_points_on_device | c.are_results_on_device)
    return eIcicleError::INVALID_DEVICE; // COMMENT maybe requires policy change given the possibility of multiple
                                         // devices on one machine
  if (c.is_async) return eIcicleError::INVALID_DEVICE; // TODO add support
  // FIXME fill non-implemented features from MSMConfig
  return eIcicleError::SUCCESS;
}

// Pipenger
template <typename Point>
eIcicleError cpu_msm_single_thread(
  const Device& device,
  const scalar_t* scalars, // COMMENT it assumes no negative scalar inputs
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  Point* results)
{
  auto t = Timer("total-msm");
  // TODO remove at the end
  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext->get<int>("c"); // TODO calculate instead of param
  const unsigned int precomp_f = config.precompute_factor;
  const int num_bms = ((scalar_t::NBITS - 1) / (precomp_f * c)) + 1;

  Point* bkts = msm_bucket_accumulator<Point>(
    scalars, bases, c, num_bms, msm_size, precomp_f, config.are_scalars_montgomery_form,
    config.are_points_montgomery_form);
  Point* bm_sums = msm_bm_sum<Point>(bkts, c, num_bms);
  Point res = msm_final_sum<Point>(bm_sums, c, num_bms, config.are_points_montgomery_form);

  results[0] = res;
  msm_delete_arrays(bkts, bm_sums, num_bms);
  return eIcicleError::SUCCESS;
}

template <typename Point>
Point int_mult(Point p, int x)
{
  if (Point::is_zero(p) || x == 0) return Point::zero();

  Point result = Point::zero();
  if (x & 1) result = p;
  x >>= 1;
  while (x > 0) {
    p = p + p;
    if (x & 1) result = result + p;
    x >>= 1;
  }
  return result;
}

template <typename Point>
Msm<Point>::Msm(const MSMConfig& config)
    : n_threads(config.ext->get<int>("n_threads")),
      c(config.ext->get<int>("c")), // TODO calculate instead of param
      precomp_f(config.precompute_factor), num_bms(((scalar_t::NBITS - 1) / (config.precompute_factor * c)) + 1),
      are_scalars_mont(config.are_scalars_montgomery_form), are_points_mont(config.are_points_montgomery_form),
      manager(config.ext->get<int>("n_threads")),
#ifdef DEBUG_PRINTS
      trace_f("trace_bucket_multi.txt"), // TODO delete
#endif
      num_bkts(1 << (c - 1)),
      log_num_segments(std::max((int)std::ceil(std::log2(n_threads * TASKS_PER_THREAD / (2 * num_bms))), 0)),
      num_bm_segments(1 << log_num_segments), segment_size(num_bkts >> log_num_segments)
{
  // Phase 1
  bkts = new Point[num_bms * num_bkts];
  std::fill_n(bkts, num_bkts * num_bms, Point::zero()); // TODO remove as occupancy removes the need of initial value
  bkts_occupancy = new bool[num_bms * num_bkts];
  std::fill_n(bkts_occupancy, num_bkts * num_bms, false);

  // Phase 2
  phase2_sums =
    new Point[num_bms * num_bm_segments * 2]; // Both triangle and line sum one after the other for each segment
  task_assigned_to_sum = new std::tuple<int, int>[num_bms * num_bm_segments * 2];
  // std::fill_n(task_assigned_to_sum, std::make_tuple(-1, -1));
  for (int i = 0; i < num_bms * num_bm_segments * 2; i++)
    task_assigned_to_sum[i] = std::make_tuple(1, 1);

  bm_sums = new Point[num_bms];

  // threads = new WorkThread<Point>[n_threads];
  // for (int i = 0; i < n_threads; i++)
  //   threads[i].thread_setup(i, tasks_per_thread);

#ifdef DEBUG_PRINTS
  if (!trace_f.good()) { throw std::invalid_argument("Can't open file"); } // TODO remove log
#endif
}

template <typename Point>
Msm<Point>::~Msm()
{
  std::cout << "\n\nDestroying msm object at the end of the run\n\n"; // COMMENT why am I not seeing it one the console?
                                                                      // Isn't the destructor automatically called when
                                                                      // msm goes out of scope?
// for (int i = 0; i < n_threads; i++) threads[i].thread.join();
#ifdef DEBUG_PRINTS
  trace_f.close();
#endif
  // delete[] threads;
  delete[] bkts;
  delete[] bkts_occupancy;

  std::cout << "Loops without a free thread:\t" << loop_count << '\n';
}

template <typename Point>
void Msm<Point>::bkt_file()
{
  bkts_f.open("buckets_multi.txt");
  if (!bkts_f.good()) { throw std::invalid_argument("Can't open file"); } // TODO remove log

  for (int i = 0; i < num_bms; i++)
    for (int j = 0; j < num_bkts; j++)
      bkts_f << '(' << i << ',' << j << "):\t" << Point::to_affine(bkts[num_bkts * i + j]).x << '\n';
  bkts_f.close();
}

template <typename Point>
void Msm<Point>::wait_for_idle()
{
  /**
   * Handle thread outputs and possible collisions between results and occupied bkts
   * @param threads - working threads array
   * @param n_threads - size of the above array
   * @param bkts - bucket array to store results / add stored value with result
   * @param bkts_occupancy - boolean array holding the bkts_occupancy of each bucket
   * @return boolean, a flag indicating all threads are idle
   */
#ifdef DEBUG_PRINTS
  trace_f << "\n\n=#=#=#=# Wait for idle =#=#=#=#\n\n";
#endif

  ECaddTask<Point, Point>* task = nullptr;
  manager.get_completed_task(task);
  int count = 0;
  while (task != nullptr)
  {
    // std::cout << count << ":\tNew completed task.\tstatus=" << task->status << " (Idle = " << task->is_idle() << ")" << "\n";
    count++;
    if (bkts_occupancy[task->return_idx])
    {
      task->p1 = task->result;
      task->p2 = bkts[task->return_idx];
      bkts_occupancy[task->return_idx] = false;
#ifdef DEBUG_PRINTS
      trace_f << '#' << task->return_idx << ":\tFCollision addition - bkts' cell:\t"
              << Point::to_affine(bkts[task->return_idx]).x << "\t(With add res point:\t"
              << Point::to_affine(task->result).x << " = " << Point::to_affine(bkts[task->return_idx] + task->result).x
              << ")\t(" << task << ")\n";
#endif
      task->dispatch();
    }
    else
    {
      bkts[task->return_idx] = task->result;
      bkts_occupancy[task->return_idx] = true;
#ifdef DEBUG_PRINTS
      trace_f << '#' << task->return_idx << ":\tFRes write free cell:\t" << task->result << '\n';
#endif
    }
    manager.get_completed_task(task);
  }
}

template <typename Point>
void Msm<Point>::phase1_push_addition(
  const unsigned int task_bkt_idx,
  const Point bkt,
  const Point& base,
  int pidx) // TODO add option of adding different types
{
  /**
   * Assign EC addition to a free thread
   * @param task_bkt_idx - results address in the bkts array
   * @param bkt - bkt to be added. it is passed by value to allow the appropriate cell in the bucket array to be "free"
   * an overwritten without affecting the working thread
   * @param p2 - point to be added
   */
  ECaddTask<Point, Point>* task = nullptr; // TODO actually use the 2 types according to the phase and remove templates
  while (task == nullptr) 
  {
    bool holds_result = manager.get_free_task(task);
    if (holds_result)
    {
      if (bkts_occupancy[task->return_idx])
      {
        task->p1 = task->result;
        task->p2 = bkts[task->return_idx];
        bkts_occupancy[task->return_idx] = false;
#ifdef DEBUG_PRINTS
        trace_f << '#' << task->return_idx << ":\tCollision addition - bkts' cell:\t"
                << Point::to_affine(bkts[task->return_idx]).x << "\t(With add res point:\t"
                << Point::to_affine(task->result).x << " = " << Point::to_affine(bkts[task->return_idx] + task->result).x
                << ")\t(" << task << ")\n";
#endif
        task->dispatch();
        task = nullptr;
        continue;
      }
      else 
      {
        bkts[task->return_idx] = task->result;
        bkts_occupancy[task->return_idx] = true;
#ifdef DEBUG_PRINTS
        trace_f << '#' << task->return_idx << ":\tRes write free cell:\t" << task->result << '\n';
#endif
      }
    }
    task->p1 = bkt;
    task->p2 = base;
    task->return_idx = task_bkt_idx;
#ifdef DEBUG_PRINTS
    trace_f << '#' << task_bkt_idx << ":\tAssigned to:\t(" << task << ")\n";
#endif
    task->dispatch();
  }
}

template <typename Point>
Point* Msm<Point>::bucket_accumulator(const scalar_t* scalars, const affine_t* bases, const unsigned int msm_size)
{
  // TODO In class function definition
  /**
   * Accumulate into the different bkts
   * @param scalars - original scalars given from the msm result
   * @param bases - point bases to add
   * @param msm_size - number of scalars to add
   * @return bkts - points array containing all bkts
   */
  auto t = Timer("P1:bucket-accumulator");
  uint32_t coeff_bit_mask = num_bkts - 1;
  const int num_windows_m1 = (scalar_t::NBITS - 1) / c; // +1 for ceiling than -1 for m1
  int carry = 0;

  std::cout << "\n\nc=" << c << "\tpcf=" << precomp_f << "\tnum bms=" << num_bms << "\tntrds,tasks=" << n_threads << "\n\n\n";
  std::cout << log_num_segments << '\n' << segment_size << "\n\n\n";
  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    scalar_t scalar = are_scalars_mont ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(scalar_t::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = scalar_t::neg(scalar);
    for (int j = 0; j < precomp_f; j++) {
      affine_t point = are_points_mont ? affine_t::from_montgomery(bases[precomp_f * i + j])
                                       : bases[precomp_f * i + j]; // TODO change here order of precomp
      if (negate_p_and_s) point = affine_t::neg(point);
      for (int k = 0; k < num_bms; k++) {
        if (num_bms * j + k > num_windows_m1) { break; } // Avoid seg fault in case precomp_f*c exceeds the scalar width

        uint32_t curr_coeff = scalar.get_scalar_digit(num_bms * j + k, c) + carry;
        int bkt_idx = 0;
        if ((curr_coeff & ((1 << c) - 1)) != 0) {
          if (curr_coeff < num_bkts) {
            bkt_idx = num_bkts * k + curr_coeff;
            carry = 0;
          } else {
            bkt_idx = num_bkts * k + ((-curr_coeff) & coeff_bit_mask);
            carry = 1;
          }
          if (bkts_occupancy[bkt_idx]) {
            bkts_occupancy[bkt_idx] = false;
#ifdef DEBUG_PRINTS
            trace_f << '#' << bkt_idx << ":\tRead for addition:\t" << Point::to_affine(bkts[bkt_idx]).x
                    << "\t(With new point:\t" << (carry > 0 ? affine_t::neg(point) : point).x << " = "
                    << Point::to_affine(bkts[bkt_idx] + (carry > 0 ? affine_t::neg(point) : point)).x << ")\n";
// trace_f.flush();
#endif
            phase1_push_addition(bkt_idx, bkts[bkt_idx], carry > 0 ? Point::from_affine(affine_t::neg(point)) : Point::from_affine(point), i);
          } else {
            bkts_occupancy[bkt_idx] = true;
            bkts[bkt_idx] = carry > 0 ? Point::neg(Point::from_affine(point)) : Point::from_affine(point);
#ifdef DEBUG_PRINTS
            trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << (carry > 0 ? affine_t::neg(point) : point).x
                    << '\n';
#endif
          }
        } else
          carry = curr_coeff >> c; // Edge case for coeff = 1 << c due to carry overflow
      }
    }
  }
  std::cout << "Wait for idle\n";
  wait_for_idle();
#ifdef DEBUG_PRINTS
  bkt_file();
#endif
  return bkts;
}

// template <typename Point>
// std::tuple<int, int>
// Msm<Point>::phase2_push_addition(const unsigned int task_bkt_idx, const Point& bkt, const Point& base);
// {
//   return std::make_tuple(-1, -1);
// }

template <typename Point>
Point* Msm<Point>::bm_sum()
{
  /**
   * Sum bucket modules to one point each
   * @param bkts - point array containing all bkts ordered by bucket module
   * @return bm_sums - point array containing the bucket modules' sums
   */
  // Init values of partial (line) and total (triangle) sum
  for (int i = 0; i < num_bms; i++) {
    for (int j = 0; j < num_bm_segments - 1; j++) {
      phase2_sums[num_bm_segments * i + 2 * j] =
        bkts[num_bkts * i + segment_size * (j + 1)]; // +1 because the sum starts from the most significant element of
                                                     // the segment
      phase2_sums[num_bm_segments * i + 2 * j + 1] = bkts[num_bkts * i + segment_size * (j + 1)];
    }
    phase2_sums[num_bm_segments * (i + 1) - 2] =
      bkts[num_bkts * i]; // The most significant bucket of every bm is stored in address 0
    phase2_sums[num_bm_segments * (i + 1) - 1] = bkts[num_bkts * i];
  }

  for (int k = segment_size - 1; k > 0; k--) {
    for (int i = 0; i < num_bms; i++) {
      for (int j = 0; j < num_bm_segments; j++) {
        // For readability
        int triangle_sum_idx = num_bm_segments * i + 2 * j;
        int line_sum_idx = num_bm_segments * i + 2 * j + 1;
        int bkt_idx = num_bkts * i + segment_size * j + k;
        int assigned_thread = std::get<0>(task_assigned_to_sum[line_sum_idx]);
        int assigned_task = std::get<1>(task_assigned_to_sum[line_sum_idx]);
        if (assigned_thread >= 0) {
          // while (!threads[assigned_thread].task[assigned_task].out_done.load(std::memory_order_acquire)) {
          // } // TODO add sleep
          // task_assigned_to_sum[]
        }

        phase2_push_addition(line_sum_idx, phase2_sums[line_sum_idx], bkts[bkt_idx]);
      }
    }
  }
}

template <typename Point>
eIcicleError cpu_msm(
  const Device& device,
  const scalar_t* scalars, // COMMENT it assumes no negative scalar inputs
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  Point* results)
{
  auto msm = new Msm<Point>(config);
  // TODO move the function here instead of redundant call
  // return cpu_msm_multithreaded<Point>(device, msm, scalars, bases, msm_size, config, results);
  auto t = Timer("total-msm");
  // TODO remove at the end
  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext->get<int>("c"); // TODO calculate instead of param
  const unsigned int precomp_f = config.precompute_factor;
  const int num_bms = ((scalar_t::NBITS - 1) / (precomp_f * c)) + 1;

  if (config.ext->get<int>("n_threads") <= 0) { return eIcicleError::INVALID_ARGUMENT; }

  Point* bkts = msm->bucket_accumulator(scalars, bases, msm_size);
  // Point* bm_sums = msm->bm_sum(bkts, c, num_bms);
  Point* bm_sums = msm_bm_sum<Point>(bkts, c, num_bms);

  Point res = msm_final_sum<Point>(bm_sums, c, num_bms, config.are_points_montgomery_form);
  results[0] = res;
  delete[] bm_sums;
  delete msm;

  // Todo mem management (smart pointer?)
  return eIcicleError::SUCCESS;
}

template <typename Point>
eIcicleError cpu_msm_ref(
  const Device& device,
  const scalar_t* scalars,
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  Point* results)
{
  const unsigned int precomp_f = config.precompute_factor;
  Point res = Point::zero();
  for (auto i = 0; i < msm_size; ++i) {
    scalar_t scalar = config.are_scalars_montgomery_form ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    affine_t point =
      config.are_points_montgomery_form ? affine_t::from_montgomery(bases[precomp_f * i]) : bases[precomp_f * i];
    res = res + scalar * Point::from_affine(point);
  }
  // results[0] = config.are_points_montgomery_form? Point::to_montgomery(res) : res;
  results[0] = res;
  return eIcicleError::SUCCESS;
}

// COMMENT should I add it to the class
template <typename A>
eIcicleError cpu_msm_precompute_bases(
  const Device& device,
  const A* input_bases,
  int nof_bases,
  const MSMConfig& config,
  A* output_bases) // Pre assigned?
{
  int precompute_factor = config.precompute_factor;
  bool is_mont = config.ext->get<bool>("is_mont");
  const unsigned int c = config.ext->get<int>("c");
  const unsigned int num_bms_no_precomp = (scalar_t::NBITS - 1) / c + 1;
  const unsigned int shift = c * ((num_bms_no_precomp - 1) / precompute_factor + 1);
  for (int i = 0; i < nof_bases; i++) {
    output_bases[precompute_factor * i] = input_bases[i]; // COMMENT Should I copy? (not by reference)
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

// BUG I think there's a memory leak somewhere as vscode crashes after a long run