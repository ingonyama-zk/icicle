#ifndef CPU_MSM
#define CPU_MSM

// #define STANDALONE
#include <atomic>
#include <mutex>
#include <tuple>

#include <unistd.h> // TODO remove

#include "icicle/errors.h"
#include "icicle/config_extension.h"
using namespace icicle;
#ifndef STANDALONE
#include "icicle/backend/msm_backend.h"
#include "icicle/runtime.h"

#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"

#include "tasks_manager.cpp"

using aff_test = affine_t;
using proj_test = projective_t;
using sca_test = scalar_t;
#else
#include <iostream>
#include <random>

#include "dummy_classes.cpp"

using aff_test = Dummy_Projective;
using proj_test = Dummy_Projective;
using sca_test = Dummy_Scalar;
#endif

#include <thread>
#include <string>
#include <iostream>
#include <fstream>

#ifndef STANDALONE
using namespace curve_config;
#endif

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
    const sca_test* scalars,
    const aff_test* bases,
    const unsigned int msm_size); // TODO change type in the end to void

  Point* bm_sum();
};

#endif