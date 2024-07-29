
#include "cpu_msm.hpp"

// #define DEBUG_PRINTS

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <atomic>

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

// TODO ask for help about memory management before / at C.R.
template <typename Point>
Point* msm_bucket_accumulator(
  const sca_test* scalars,
  const aff_test* bases,
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
  const int num_windows_m1 = (sca_test::NBITS - 1) / c;
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
    sca_test scalar = is_s_mont ? sca_test::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(sca_test::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = sca_test::neg(scalar);
    for (int j = 0; j < precomp_f; j++) {
      aff_test point = is_b_mont ? aff_test::from_montgomery(bases[precomp_f * i + j]) : bases[precomp_f * i + j];
      if (negate_p_and_s) point = aff_test::neg(point);
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
              trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << aff_test::neg(point).x << '\n';
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
  const sca_test* scalars, // COMMENT it assumes no negative scalar inputs
  const aff_test* bases,
  int msm_size,
  const MSMConfig& config,
  Point* results)
{
  auto t = Timer("total-msm");
  // TODO remove at the end
  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext.get<int>("c"); // TODO calculate instead of param
  const unsigned int precomp_f = config.precompute_factor;
  const int num_bms = ((sca_test::NBITS - 1) / (precomp_f * c)) + 1;

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
ThreadTask<Point>::ThreadTask() : return_idx(-1), p1(Point::zero()), p2(Point::zero()), result(Point::zero())
{
}

template <typename Point>
ThreadTask<Point>::ThreadTask(
  const ThreadTask<Point>& other) // TODO delete when changing to task array instead of vector
    : return_idx(-1), p1(Point::zero()), p2(Point::zero()), result(Point::zero())
{
}

template <typename Point>
void ThreadTask<Point>::run(int tid, std::vector<int>& idle_idxs, bool& kill_thread)
{
  bool rdy_status = in_ready.load(std::memory_order_acquire);
  if (!rdy_status) idle_idxs.push_back(pidx); // TODO remove when finishing debugging
  while (!rdy_status) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    rdy_status = in_ready.load(std::memory_order_acquire);
    if (kill_thread) return;
  }
  in_ready.store(false, std::memory_order_release);
  result = p1 + p2;
  out_done.store(true, std::memory_order_release);
}

template <typename Point>
void ThreadTask<Point>::new_task(const int in_idx, const Point& in_p1, const Point& in_p2)
{
  out_done.store(false, std::memory_order_release);
  return_idx = in_idx;
  p1 = in_p1; // Copied by value to not be linked to the original bucket
  p2 = in_p2;
  in_ready.store(true, std::memory_order_release);
}

template <typename Point>
void ThreadTask<Point>::chain_task(const Point in_p2)
{
  // std::unique_lock<std::mutex> temp_lock(idle_mtx);
  out_done.store(false, std::memory_order_release);
  p2 = in_p2;
  p1 = result;
  in_ready.store(true, std::memory_order_release);
  // idle.notify_one();
}

template <typename Point>
WorkThread<Point>::~WorkThread()
{
  kill_thread = true;
  thread.join();
}

template <typename Point>
void WorkThread<Point>::thread_setup(const int tid, const int task_per_thread)
{
  this->tid = tid;
  for (int j = 0; j < task_per_thread; j++)
    tasks.push_back(ThreadTask<Point>()); // TODO change to array
  thread = std::thread(
    &WorkThread<Point>::add_ec_tasks, this, std::ref(kill_thread)); // TODO kill_thread is accessible from this
}

template <typename Point>
void WorkThread<Point>::add_ec_tasks(bool& kill_thread)
{
  while (!kill_thread) {
    int i = 0;
    for (ThreadTask<Point>& task : tasks) {
      task.run(tid, idle_idxs, kill_thread);
#ifdef DEBUG_PRINTS
      if (tid == 2) std::cout << i << ", bkt_idx=" << tasks[task_round_robin].return_idx << "\tDone\n";
      i++;
#endif
    }
  }
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
    : n_threads(config.ext.get<int>("n_threads")),
      // tasks_per_thread(config.ext.get<int>("tasks_per_thread")), // TODO add tp config?
      c(config.ext.get<int>("c")), // TODO calculate instead of param
      precomp_f(config.precompute_factor), num_bms(((sca_test::NBITS - 1) / (config.precompute_factor * c)) + 1),
      are_scalars_mont(config.are_scalars_montgomery_form), are_points_mont(config.are_points_montgomery_form),
      kill_threads(false),
#ifdef DEBUG_PRINTS
      trace_f("trace_bucket_multi.txt"), // TODO delete
#endif
      thread_round_robin(0), num_bkts(1 << (c - 1)),
      log_num_segments(std::max((int)std::ceil(std::log2(num_bms / n_threads)), 0)),
      num_bm_segments(1 << log_num_segments), segment_size(num_bkts >> log_num_segments),
      tasks_per_thread(std::max(((int)(num_bms / n_threads)) << (log_num_segments + 1), 2))
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

  threads = new WorkThread<Point>[n_threads];
  for (int i = 0; i < n_threads; i++)
    threads[i].thread_setup(i, tasks_per_thread);

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
  kill_threads = true; // TODO check if it is used after kill_thread has been added to WorkThread destructor
// for (int i = 0; i < n_threads; i++) threads[i].thread.join();
#ifdef DEBUG_PRINTS
  trace_f.close();
#endif
  delete[] threads;
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
  bool all_threads_idle = false;
  int count = 0;
  while (!all_threads_idle) {
    all_threads_idle = true;

    for (int i = 0; i < n_threads; i++) {
      int og_task_round_robin = threads[i].task_round_robin;

#ifdef DEBUG_PRINTS
      trace_f << "Finishing thread " << i << ", starting at task: " << og_task_round_robin << '\n';
#endif

      for (int j = og_task_round_robin; j < og_task_round_robin + tasks_per_thread; j++) {
        int task_idx = j;
        if (task_idx >= tasks_per_thread) task_idx -= tasks_per_thread;

        // For readability
        ThreadTask<Point>& task = threads[i].tasks[task_idx];
        if (task.out_done.load(std::memory_order_acquire)) {
          if (task.return_idx >= 0) {
            if (bkts_occupancy[task.return_idx]) {
#ifdef DEBUG_PRINTS
              trace_f << '#' << task.return_idx << ":\tFCollision addition - bkts' cell:\t"
                      << Point::to_affine(bkts[task.return_idx]).x << "\t(With add res point:\t"
                      << Point::to_affine(task.result).x << " = "
                      << Point::to_affine(bkts[task.return_idx] + task.result).x << ")\t(" << i << ',' << task_idx
                      << ")\n";
#endif
              // std::cout << "\n" << i << ":\t(" << task_idx << "->" << threads[i].task_round_robin <<
              // ")\tChaining\n\n";
              bkts_occupancy[task.return_idx] = false;
              int bkt_idx = task.return_idx;
              task.return_idx = -1;

              threads[i].tasks[threads[i].task_round_robin].new_task(bkt_idx, task.result, bkts[bkt_idx]);
              if (threads[i].task_round_robin == tasks_per_thread - 1)
                threads[i].task_round_robin = 0;
              else
                threads[i].task_round_robin++;

              all_threads_idle = false; // This thread isn't idle due to the newly assigned task
            } else {
              bkts[task.return_idx] = task.result;
#ifdef DEBUG_PRINTS
              trace_f << '#' << task.return_idx << ":\tFWrite (res) free cell:\t"
                      << Point::to_affine(bkts[task.return_idx]).x << "\t(" << i << ',' << task_idx << ")\n";
#endif
              bkts_occupancy[task.return_idx] = true;
              task.return_idx = -1; // To ensure no repeated handling of outputs
            }
          } else {
#ifdef DEBUG_PRINTS
            trace_f << "Task " << task_idx << " idle\n";
#endif
          }
        } else {
          all_threads_idle = false;
          break;
#ifdef DEBUG_PRINTS
          trace_f << '#' << task.return_idx << ":\t(" << i << ',' << task_idx << ") not done\tres=" << task.result
                  << "\tstatus:" << task.in_ready << ',' << task.out_done << '\n';
#endif
        }
      }
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }
  // trace_f.flush();
}

template <typename Point>
template <typename Base>
void Msm<Point>::phase1_push_addition(
  const unsigned int task_bkt_idx,
  const Point bkt,
  const Base& base,
  int pidx) // TODO add option of adding different types
{
  /**
   * Assign EC addition to a free thread
   * @param task_bkt_idx - results address in the bkts array
   * @param bkt - bkt to be added. it is passed by value to allow the appropriate cell in the bucket array to be "free"
   * an overwritten without affecting the working thread
   * @param p2 - point to be added
   */
  int count_thread_iters = 0;
  bool assigned_to_thread = false;
  while (!assigned_to_thread) {
    // For readability
    ThreadTask<Point>& task = threads[thread_round_robin].tasks[threads[thread_round_robin].task_round_robin];
    if (task.out_done.load(std::memory_order_acquire)) {
      num_additions++;
      if (task.return_idx >= 0) {
        if (bkts_occupancy[task.return_idx]) {
#ifdef DEBUG_PRINTS
          trace_f << '#' << task.return_idx << ":\tCollision addition - bkts' cell:\t"
                  << Point::to_affine(bkts[task.return_idx]).x << "\t(With add res point:\t"
                  << Point::to_affine(task.result).x << " = " << Point::to_affine(bkts[task.return_idx] + task.result).x
                  << ")\t(" << thread_round_robin << ',' << threads[thread_round_robin].task_round_robin << ")\n";
#endif
          bkts_occupancy[task.return_idx] = false;
          ;
          task.pidx = pidx;
          task.chain_task(bkts[task.return_idx]);
        } else {
#ifdef DEBUG_PRINTS
          trace_f << '#' << task.return_idx << ":\tWrite (res) free cell:\t" << Point::to_affine(task.result).x << "\t("
                  << thread_round_robin << ',' << threads[thread_round_robin].task_round_robin << ")\n";
#endif
          bkts[task.return_idx] = task.result;
          bkts_occupancy[task.return_idx] = true;
          task.pidx = pidx;
          task.new_task(task_bkt_idx, bkt, Point::from_affine(base)); // TODO support multiple types
          assigned_to_thread = true;
#ifdef DEBUG_PRINTS
          trace_f << '#' << task_bkt_idx << ":\tAssigned to:\t(" << thread_round_robin << ','
                  << threads[thread_round_robin].task_round_robin << ")\n";
// trace_f.flush();
#endif
          // break;
        }
      } else {
        task.pidx = pidx;
        task.new_task(task_bkt_idx, bkt, Point::from_affine(base)); // TODO support multiple types
        assigned_to_thread = true;
#ifdef DEBUG_PRINTS
        trace_f << '#' << task_bkt_idx << ":\tAssigned to:\t(" << thread_round_robin << ','
                << threads[thread_round_robin].task_round_robin << ")\n";
// trace_f.flush();
#endif
        // break;
      }
      if (threads[thread_round_robin].task_round_robin == tasks_per_thread - 1)
        threads[thread_round_robin].task_round_robin = 0;
      else
        threads[thread_round_robin].task_round_robin++;
    }

    // Move to next thread after checking all current thread's tasks
    if (thread_round_robin == n_threads - 1)
      thread_round_robin = 0;
    else
      thread_round_robin++;
    count_thread_iters++;
    if (count_thread_iters == n_threads) {
      count_thread_iters = 0;
      loop_count++;
    }
  }
}

template <typename Point>
Point* Msm<Point>::bucket_accumulator(const sca_test* scalars, const aff_test* bases, const unsigned int msm_size)
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
  const int num_windows_m1 = (sca_test::NBITS - 1) / c; // +1 for ceiling than -1 for m1
  int carry = 0;

  std::cout << "\n\nc=" << c << "\tpcf=" << precomp_f << "\tnum bms=" << num_bms << "\tntrds,tasks=" << n_threads << ','
            << tasks_per_thread << "\n\n\n";
  std::cout << log_num_segments << '\n' << segment_size << "\n\n\n";
  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    sca_test scalar = are_scalars_mont ? sca_test::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(sca_test::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = sca_test::neg(scalar);
    for (int j = 0; j < precomp_f; j++) {
      aff_test point = are_points_mont ? aff_test::from_montgomery(bases[precomp_f * i + j])
                                       : bases[precomp_f * i + j]; // TODO change here order of precomp
      if (negate_p_and_s) point = aff_test::neg(point);
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
                    << "\t(With new point:\t" << (carry > 0 ? aff_test::neg(point) : point).x << " = "
                    << Point::to_affine(bkts[bkt_idx] + (carry > 0 ? aff_test::neg(point) : point)).x << ")\n";
// trace_f.flush();
#endif
            phase1_push_addition<aff_test>(bkt_idx, bkts[bkt_idx], carry > 0 ? aff_test::neg(point) : point, i);
          } else {
            bkts_occupancy[bkt_idx] = true;
            bkts[bkt_idx] = carry > 0 ? Point::neg(Point::from_affine(point)) : Point::from_affine(point);
#ifdef DEBUG_PRINTS
            trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << (carry > 0 ? aff_test::neg(point) : point).x
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

template <typename Point>
std::tuple<int, int>
Msm<Point>::phase2_push_addition(const unsigned int task_bkt_idx, const Point& bkt, const Point& base)
{
  return std::make_tuple(-1, -1);
}

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
          while (!threads[assigned_thread].task[assigned_task].out_done.load(std::memory_order_acquire)) {
          } // TODO add sleep
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
  const sca_test* scalars, // COMMENT it assumes no negative scalar inputs
  const aff_test* bases,
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

  const unsigned int c = config.ext.get<int>("c"); // TODO calculate instead of param
  const unsigned int precomp_f = config.precompute_factor;
  const int num_bms = ((sca_test::NBITS - 1) / (precomp_f * c)) + 1;

  if (config.ext.get<int>("n_threads") <= 0) { return eIcicleError::INVALID_ARGUMENT; }

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
  const sca_test* scalars,
  const aff_test* bases,
  int msm_size,
  const MSMConfig& config,
  Point* results)
{
  const unsigned int precomp_f = config.precompute_factor;
  Point res = Point::zero();
  for (auto i = 0; i < msm_size; ++i) {
    sca_test scalar = config.are_scalars_montgomery_form ? sca_test::from_montgomery(scalars[i]) : scalars[i];
    aff_test point =
      config.are_points_montgomery_form ? aff_test::from_montgomery(bases[precomp_f * i]) : bases[precomp_f * i];

    // std::cout << scalar << "\t*\t" << point << '\n';
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
  int precompute_factor,
  const MsmPreComputeConfig& config,
  A* output_bases) // Pre assigned?
{
  bool is_mont = config.ext.get<bool>("is_mont");
  const unsigned int c = config.ext.get<int>("c");
  const unsigned int num_bms_no_precomp = (sca_test::NBITS - 1) / c + 1;
  const unsigned int shift = c * ((num_bms_no_precomp - 1) / precompute_factor + 1);
  // std::cout << "Starting precompute\n";
  // std::cout << c << ',' << shift << '\n';
  for (int i = 0; i < nof_bases; i++) {
    output_bases[precompute_factor * i] = input_bases[i]; // COMMENT Should I copy? (not by reference)
    proj_test point = proj_test::from_affine(is_mont ? A::from_montgomery(input_bases[i]) : input_bases[i]);
    // std::cout << "OG point[" << i << "]:\t" << point << '\n';
    for (int j = 1; j < precompute_factor; j++) {
      for (int k = 0; k < shift; k++) {
        point = proj_test::dbl(point);
        // std::cout << point << '\n';
      }
      // std::cout << "Shifted point[" << i << "]:\t" << point << "\nStored in index=" << (precompute_factor*i+j) <<
      // '\n';
      output_bases[precompute_factor * i + j] = is_mont
                                                  ? A::to_montgomery(proj_test::to_affine(point))
                                                  : proj_test::to_affine(point); // TODO change here order of precomp
    }
  }
  return eIcicleError::SUCCESS;
}

// BUG I think there's a memory leak somewhere as vscode crashes after a long run

#ifndef STANDALONE

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<aff_test>);
REGISTER_MSM_BACKEND("CPU", cpu_msm<proj_test>);
// REGISTER_MSM_BACKEND("CPU", cpu_msm_single_thread<proj_test>);

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU_REF", cpu_msm_precompute_bases<aff_test>);
// REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_ref<proj_test>); // TODO revert to yuval's ref when testing batched msm
REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_single_thread<proj_test>);
#else

static MSMConfig default_msm_config()
{
  MSMConfig config = {
    0,     // nof_bases
    1,     // precompute_factor
    1,     // batch_size
    false, // are_scalars_on_device
    false, // are_scalars_montgomery_form
    false, // are_points_on_device
    false, // are_points_montgomery_form
    false, // are_results_on_device
    false, // is_async
  };
  // TODO: maybe allow backends to register default values and call it here so they can fill the ext
  config.ext.set("c", 0);
  config.ext.set("bitsize", 0);
  config.ext.set("large_bucket_factor", 10);
  config.ext.set("big_triangle", true);
  return config;
}

static MsmPreComputeConfig default_msm_pre_compute_config()
{
  MsmPreComputeConfig config = {
    false, // is_input_on_device
    false, // is_output_on_device
    false, // is_async
  };
  // TODO: maybe allow backends to register default values and call it here so they can fill the ext
  config.ext.set("c", 0);
  return config;
}

int main()
{
  int seed = 0;
  auto t = Timer("Time till failure");

  while (true) {
    const int logn = 5;
    const int N = 1 << logn;
    auto scalars = std::make_unique<sca_test[]>(N);
    auto bases = std::make_unique<aff_test[]>(N);

    bool conv_mont = false;

    std::mt19937_64 generator(seed);
    sca_test::rand_host_many(scalars.get(), N, generator);
    proj_test::rand_host_many_affine(bases.get(), N, generator);
    if (conv_mont) {
      for (int i = 0; i < N; i++)
        bases[i] = aff_test::to_montgomery(bases[i]);
    }
    proj_test result_cpu{};
    proj_test result_cpu_dbl_n_add{};
    proj_test result_cpu_ref{};

    proj_test result{};

    auto run = [&](const char* dev_type, proj_test* result, const char* msg, bool measure, int iters, auto cpu_msm) {
      const int log_p = 3;
      const int c = std::max(logn, 8) - 1;
      const int pcf = 1 << log_p;

      int hw_threads = std::thread::hardware_concurrency();
      if (hw_threads <= 0) { std::cout << "Unable to detect number of hardware supported threads - fixing it to 1\n"; }
      // const int n_threads = (hw_threads > 1)? hw_threads-2 : 1;
      const int n_threads = 8;

      const int tasks_per_thread = 4;

      auto config = default_msm_config();
      config.ext.set("c", c);
      config.ext.set("n_threads", n_threads);
      config.ext.set("tasks_per_thread", tasks_per_thread);
      config.precompute_factor = pcf;
      config.are_scalars_montgomery_form = false;
      config.are_points_montgomery_form = conv_mont;

      auto pc_config = default_msm_pre_compute_config();
      pc_config.ext.set("c", c);
      pc_config.ext.set("is_mont", config.are_points_montgomery_form);

      auto precomp_bases = std::make_unique<aff_test[]>(N * pcf);
      cpu_msm_precompute_bases<aff_test>({}, bases.get(), N, pcf, pc_config, precomp_bases.get());
      // START_TIMER(MSM_sync)
      for (int i = 0; i < iters; ++i) {
        // TODO real test
        // msm_precompute_bases(bases.get(), N, 1, default_msm_pre_compute_config(), bases.get());
        cpu_msm({}, scalars.get(), precomp_bases.get(), N, config, result);
      }
      // END_TIMER(MSM_sync, msg, measure);
    };

    // run("CPU", &result_cpu_dbl_n_add, "CPU msm", false /*=measure*/, 1 /*=iters*/); // warmup
    run("CPU", &result_cpu, "CPU msm", true /*=measure*/, 1 /*=iters*/, cpu_msm<proj_test>);
    run("CPU_REF", &result_cpu_ref, "CPU_REF msm", true /*=measure*/, 1 /*=iters*/, cpu_msm_single_thread<proj_test>);
    std::cout << proj_test::to_affine(result_cpu) << std::endl;
    std::cout << proj_test::to_affine(result_cpu_ref) << std::endl;
    std::cout << "Seed is: " << seed << '\n';
    assert(result_cpu == result_cpu_ref);
  }

  return 0;
}
#endif