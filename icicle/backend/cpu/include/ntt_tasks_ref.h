#pragma once
#include "icicle/backend/ntt_backend.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/fields/field_config.h"
#include "icicle/vec_ops.h"
#include "tasks_manager.h"
#include "cpu_ntt_domain.h"

#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>

using namespace field_config;
using namespace icicle;
namespace ntt_cpu {

  constexpr uint32_t layers_sub_logn_ref[31][3] = {
    // {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {1, 1, 1},   {2, 1, 1},   {2, 2, 1},   {3, 2, 1},   {4, 3, 0},
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

  struct NttTaskCordinatesRef {
    int h1_layer_idx = 0;
    int h1_subntt_idx = 0;
    int h0_layer_idx = 0;
    int h0_block_idx = 0;
    int h0_subntt_idx = 0;

    // Comparison operators for map
    bool operator<(const NttTaskCordinatesRef& other) const
    {
      return std::tie(h1_layer_idx, h1_subntt_idx, h0_layer_idx, h0_block_idx, h0_subntt_idx) <
             std::tie(
               other.h1_layer_idx, other.h1_subntt_idx, other.h0_layer_idx, other.h0_block_idx, other.h0_subntt_idx);
    }
  };

  struct NttSubLognRef {
    int logn;                                         // Original size of the problem
    std::vector<std::vector<int>> h0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<int> h1_layers_sub_logn;              // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubLognRef(int logn) : logn(logn)
    {
      if (logn > 15) {
        // Initialize h1_layers_sub_logn
        h1_layers_sub_logn =
          std::vector<int>(std::begin(layers_sub_logn_ref[logn]), std::end(layers_sub_logn_ref[logn]));
        // Initialize h0_layers_sub_logn
        h0_layers_sub_logn = {
          std::vector<int>(
            std::begin(layers_sub_logn_ref[h1_layers_sub_logn[0]]),
            std::end(layers_sub_logn_ref[h1_layers_sub_logn[0]])),
          std::vector<int>(
            std::begin(layers_sub_logn_ref[h1_layers_sub_logn[1]]),
            std::end(layers_sub_logn_ref[h1_layers_sub_logn[1]]))};
      } else {
        h1_layers_sub_logn = {0, 0, 0};
        h0_layers_sub_logn = {
          std::vector<int>(std::begin(layers_sub_logn_ref[logn]), std::end(layers_sub_logn_ref[logn])), {0, 0, 0}};
      }
      // ICICLE_LOG_DEBUG << "NttTaskInfo: h1_layers_sub_logn: " << h1_layers_sub_logn[0] << ", " <<
      // h1_layers_sub_logn[1] << ", " << h1_layers_sub_logn[2]; ICICLE_LOG_DEBUG << "NttTaskInfo:
      // h0_layers_sub_logn[0]: " << h0_layers_sub_logn[0][0] << ", " << h0_layers_sub_logn[0][1] << ", " <<
      // h0_layers_sub_logn[0][2]; ICICLE_LOG_DEBUG << "NttTaskInfo: h0_layers_sub_logn[1]: " <<
      // h0_layers_sub_logn[1][0] << ", " << h0_layers_sub_logn[1][1] << ", " << h0_layers_sub_logn[1][2];
    }
  };

  template <typename S, typename E>
  class NttTaskRef;

  template <typename S, typename E>
  class NttTasksManagerRef;

  template <typename S = scalar_t, typename E = scalar_t>
  class NttCpuRef
  {
  public:
    NttSubLognRef ntt_sub_logn;
    NTTDir direction;
    const NTTConfig<S>& config;
    int domain_max_size;
    const S* twiddles;
    // double duration_total=0;

    NttCpuRef(int logn, NTTDir direction, const NTTConfig<S>& config, int domain_max_size, const S* twiddles)
        : ntt_sub_logn(logn), direction(direction), config(config), domain_max_size(domain_max_size), twiddles(twiddles)
    {
    }

    int bit_reverse(int n, int logn);
    uint64_t idx_in_mem(NttTaskCordinatesRef ntt_task_cordinates, int element);
    eIcicleError reorder_by_bit_reverse(NttTaskCordinatesRef ntt_task_cordinates, E* elements, bool is_top_hirarchy);
    void dit_ntt(E* elements, NttTaskCordinatesRef ntt_task_cordinates, const S* twiddles);
    void dif_ntt(E* elements, NttTaskCordinatesRef ntt_task_cordinates, const S* twiddles);
    eIcicleError
    coset_mul(E* elements, const S* twiddles, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset);
    eIcicleError reorder_input(E* input);
    void refactor_and_reorder(E* elements, const S* twiddles);
    eIcicleError reorder_output(E* elements, NttTaskCordinatesRef ntt_task_cordinates, bool is_top_hirarchy);
    void refactor_output_h0(E* elements, NttTaskCordinatesRef ntt_task_cordinates, const S* twiddles);
    eIcicleError
    h1_cpu_ntt(E* input, NttTaskCordinatesRef ntt_task_cordinates, NttTasksManagerRef<S, E>& ntt_tasks_manager);
    eIcicleError h0_cpu_ntt(E* input, NttTaskCordinatesRef ntt_task_cordinates);
    eIcicleError handle_pushed_tasks(
      TasksManager<NttTaskRef<S, E>>* tasks_manager, NttTasksManagerRef<S, E>& ntt_tasks_manager, int h1_layer_idx);
  };

  class TasksDependenciesCountersRef
  {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCountersRef(NttSubLognRef ntt_sub_logn, int h1_layer_idx);

    // Function to decrement the counter for a given task and check if it is ready to execute. if so, return true
    bool decrement_counter(NttTaskCordinatesRef ntt_task_cordinates);
    int get_nof_pointing_to_counter(int h0_layer_idx) { return nof_pointing_to_counter[h0_layer_idx]; }
    int get_nof_h0_layers() { return nof_h0_layers; }

  private:
    int h1_layer_idx;
    int nof_h0_layers;
    std::vector<int> nof_pointing_to_counter; // Number of counters for each layer

    // Each h1_subntt has its own set of counters
    std::vector<std::vector<std::vector<std::shared_ptr<int>>>>
      h0_counters; // [h1_subntt_idx][h0_layer_idx][h0_counter_idx]

    // One counter for each h1_subntt to signal the end of the h1_subntt. each h0_subntt of last h0_layer will decrement
    // this counter when it finishes and when it reaches 0, the h1_subntt is ready to reorder
    std::vector<std::shared_ptr<int>> h1_counters; // [h1_subntt_idx]
  };

  template <typename S = scalar_t, typename E = scalar_t>
  struct NttTaskParamsRef {
    NttCpuRef<S, E>* ntt_cpu;
    E* input;
    NttTaskCordinatesRef task_c;
    bool reorder;
  };

  template <typename S = scalar_t, typename E = scalar_t>
  class NttTaskRef : public TaskBase
  {
  public:
    // Default constructor
    NttTaskRef() : ntt_cpu(nullptr), input(nullptr), reorder(false) {}

    // // Constructor with parameters
    // NttTaskRef(
    //   NttCpuRef<S, E>& ntt_cpu,   // Reference to an NttCpuRef instance
    //   E* input,
    //   NttTaskCordinatesRef ntt_task_cordinates,
    //   bool reorder = false)
    //   : ntt_cpu(ntt_cpu), input(input), ntt_task_cordinates(ntt_task_cordinates), reorder(reorder) {}

    void execute()
    {
      // ICICLE_LOG_INFO << "Executing NttTaskRef";
      // auto start_h0_cpu_ntt = std::chrono::high_resolution_clock::now();
      // auto end_h0_cpu_ntt = std::chrono::high_resolution_clock::now();

      // auto thread_id = std::this_thread::get_id();  // Get the thread id
      if (reorder) {
        // if all h0_subntts are done, reorder the output
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing reorder start: h0_layer_idx: " <<
        // ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ",
        // h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
        if (ntt_cpu->config.columns_batch) {
          ntt_cpu->reorder_output(input, ntt_task_cordinates, false);
        } else {
          for (int b = 0; b < ntt_cpu->config.batch_size; b++) {
            ntt_cpu->reorder_output(input + b * (1 << (ntt_cpu->ntt_sub_logn.logn)), ntt_task_cordinates, false);
          }
        }
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing reorder done:  h0_layer_idx: " <<
        // ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ",
        // h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
      } else {
        // Execute the h0_cpu_ntt using the provided NttCpuRef instance
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing ntt start: h0_layer_idx: " <<
        // ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ",
        // h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx; Start timing the for loop for h1_cpu_ntt
        // start_h0_cpu_ntt = std::chrono::high_resolution_clock::now();
        ntt_cpu->h0_cpu_ntt(input, ntt_task_cordinates);
        // end_h0_cpu_ntt = std::chrono::high_resolution_clock::now();
        // ntt_cpu->duration_total += std::chrono::duration<double, std::milli>(end_h0_cpu_ntt -
        // start_h0_cpu_ntt).count(); ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing ntt done: h0_layer_idx:
        // " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ",
        // h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx; for (int i = 0; i < 1<<ntt_cpu->ntt_sub_logn.logn;
        // i++) {
        //   ICICLE_LOG_DEBUG << "Thread " << thread_id << ": input[" << i << "]: " << input[i];
        // }
      }
    }

    NttTaskCordinatesRef get_coordinates() const { return ntt_task_cordinates; }

    bool is_reorder() const { return reorder; }
    void set_params(NttTaskParamsRef<S, E> params)
    {
      ntt_cpu = params.ntt_cpu;
      input = params.input;
      ntt_task_cordinates = params.task_c;
      reorder = params.reorder;
    }

  private:
    NttCpuRef<S, E>* ntt_cpu; // Reference to NttCpuRef instance
    E* input;
    NttTaskCordinatesRef ntt_task_cordinates;
    bool reorder;
  };

  template <typename S = scalar_t, typename E = scalar_t>
  class NttTasksManagerRef
  {
  public:
    NttTasksManagerRef(int logn);

    // Add a new task to the ntt_task_manager
    eIcicleError push_task(NttCpuRef<S, E>* ntt_cpu, E* input, NttTaskCordinatesRef task_c, bool reorder);

    // Function to get an available task to run
    bool get_available_task_to_run(
      NttTaskRef<S, E>* available_task,
      int h1_layer); // if no available task- available_task=null. if not waiting to be ready - return false

    // Set a task as completed and update dependencies
    eIcicleError set_task_as_completed(NttTaskRef<S, E>& completed_task, int nof_subntts_l2);

    bool tasks_to_do() { return !available_tasks_list.empty() || !waiting_tasks_map.empty(); }

    bool available_tasks() { return !available_tasks_list.empty(); }

    NttTaskParamsRef<S, E> get_available_task() { return available_tasks_list.front(); }

    eIcicleError erase_task_from_available_tasks_list()
    {
      available_tasks_list.erase(available_tasks_list.begin());
      return eIcicleError::SUCCESS;
    }

  private:
    std::vector<TasksDependenciesCountersRef> counters; // Dependencies counters by layer
    std::vector<NttTaskParamsRef<S, E>> available_tasks_list;
    std::map<NttTaskCordinatesRef, NttTaskParamsRef<S, E>> waiting_tasks_map;
  };

  //////////////////////////// TasksDependenciesCountersRef Implementation ////////////////////////////

  TasksDependenciesCountersRef::TasksDependenciesCountersRef(NttSubLognRef ntt_sub_logn, int h1_layer_idx)
      : h0_counters(
          1
          << ntt_sub_logn.h1_layers_sub_logn[1 - h1_layer_idx]), // nof_h1_subntts = h1_layers_sub_logn[1-h1_layer_idx].
        h1_counters(1 << ntt_sub_logn.h1_layers_sub_logn[1 - h1_layer_idx])
  { // Initialize h1_counters with N0 * N1

    nof_h0_layers =
      ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2] ? 3 : (ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1] ? 2 : 1);
    // ICICLE_LOG_DEBUG << "TasksDependenciesCountersRef: nof_h0_layers: " << nof_h0_layers;
    nof_pointing_to_counter.resize(nof_h0_layers);
    nof_pointing_to_counter[0] = 1;
    int l1_counter_size;
    int l2_counter_size;
    int l1_nof_counters;
    int l2_nof_counters;
    if (nof_h0_layers > 1) {
      // Initialize counters for layer 1 - N2 counters initialized with N1.
      nof_pointing_to_counter[1] = 1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0];
      l1_nof_counters = 1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2];
      l1_counter_size = 1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1];
    }
    if (nof_h0_layers > 2) {
      // Initialize counters for layer 2 - N0 counters initialized with N2.
      nof_pointing_to_counter[2] = 1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1];
      l2_nof_counters = 1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0];
      l2_counter_size = 1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2];
    }

    for (int h1_subntt_idx = 0; h1_subntt_idx < (1 << ntt_sub_logn.h1_layers_sub_logn[1 - h1_layer_idx]);
         ++h1_subntt_idx) {
      h0_counters[h1_subntt_idx].resize(3); // Assuming 3 layers (0, 1, 2)

      // Initialize counters for layer 0 - 1 counter1 initialized with 0.
      h0_counters[h1_subntt_idx][0].resize(1);
      h0_counters[h1_subntt_idx][0][0] = std::make_shared<int>(0); //[h1_subntt_idx][h0_layer_idx][h0_counter_idx]
      // ICICLE_LOG_DEBUG << "TasksDependenciesCountersRef: h0_counters["<<h1_subntt_idx<<"][0][0]: " <<
      // *h0_counters[h1_subntt_idx][0][0];

      if (nof_h0_layers > 1) {
        // Initialize counters for layer 1 - N2 counters initialized with N1.
        h0_counters[h1_subntt_idx][1].resize(l1_nof_counters);
        for (int counter_idx = 0; counter_idx < l1_nof_counters; ++counter_idx) {
          h0_counters[h1_subntt_idx][1][counter_idx] = std::make_shared<int>(l1_counter_size);
          // ICICLE_LOG_DEBUG << "TasksDependenciesCountersRef: h0_counters["<<h1_subntt_idx<<"][1]["<<counter_idx<<"]:
          // " << *h0_counters[h1_subntt_idx][1][counter_idx];
        }
      }
      if (nof_h0_layers > 2) {
        // Initialize counters for layer 2 - N0 counters initialized with N2.
        h0_counters[h1_subntt_idx][2].resize(l2_nof_counters);
        for (int counter_idx = 0; counter_idx < l2_nof_counters; ++counter_idx) {
          h0_counters[h1_subntt_idx][2][counter_idx] = std::make_shared<int>(l2_counter_size);
          // ICICLE_LOG_DEBUG << "TasksDependenciesCountersRef: h0_counters["<<h1_subntt_idx<<"][2]["<<counter_idx<<"]:
          // " << *h0_counters[h1_subntt_idx][2][counter_idx];
        }
      }

      // Initialize h1_counters with N0 * N1
      int h1_counter_size = nof_h0_layers == 3 ? (1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]) *
                                                   (1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1])
                            : nof_h0_layers == 2 ? (1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0])
                                                 : 0;
      h1_counters[h1_subntt_idx] = std::make_shared<int>(h1_counter_size);
      // ICICLE_LOG_DEBUG << "TasksDependenciesCountersRef: h1_counters["<<h1_subntt_idx<<"]: " <<
      // *h1_counters[h1_subntt_idx];
    }
  }

  bool TasksDependenciesCountersRef::decrement_counter(NttTaskCordinatesRef task_c)
  {
    if (nof_h0_layers == 1) { return false; }
    if (task_c.h0_layer_idx < nof_h0_layers - 1) {
      // Extract the coordinates from the task
      int counter_group_idx = task_c.h0_layer_idx == 0 ? task_c.h0_block_idx :
                                                       /*task_c.h0_layer_idx==1*/ task_c.h0_subntt_idx;

      std::shared_ptr<int>& counter_ptr = h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx + 1][counter_group_idx];
      (*counter_ptr)--;

      if (*counter_ptr == 0) { return true; }
    } else {
      // Decrement the counter for the given h1_subntt_idx
      std::shared_ptr<int>& h1_counter_ptr = h1_counters[task_c.h1_subntt_idx];
      (*h1_counter_ptr)--;

      if (*h1_counter_ptr == 0) { return true; }
    }
    return false;
  }

  //////////////////////////// NttTasksManagerRef Implementation ////////////////////////////

  template <typename S, typename E>
  NttTasksManagerRef<S, E>::NttTasksManagerRef(int logn)
      : counters(logn > 15 ? 2 : 1, TasksDependenciesCountersRef(NttSubLognRef(logn), 0))
  {
    // ICICLE_LOG_INFO << "NttTasksManagerRef constructor";
    if (logn > 15) { counters[1] = TasksDependenciesCountersRef(NttSubLognRef(logn), 1); }
  }

  template <typename S, typename E>
  eIcicleError
  NttTasksManagerRef<S, E>::push_task(NttCpuRef<S, E>* ntt_cpu, E* input, NttTaskCordinatesRef task_c, bool reorder)
  {
    // ICICLE_LOG_INFO << "NttTasksManagerRef<S, E>::push_task";
    // Create a new NttTaskParamsRef and add it to the available_tasks_list
    NttTaskParamsRef<S, E> params = {ntt_cpu, input, task_c, reorder};
    if (task_c.h0_layer_idx == 0) {
      available_tasks_list.push_back(params);
    } else {
      waiting_tasks_map[task_c] = params; // Add to map
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  bool NttTasksManagerRef<S, E>::get_available_task_to_run(NttTaskRef<S, E>* available_task, int h1_layer)
  {
    if (!available_tasks_list.empty()) {
      // Take the first task from the list
      NttTaskParamsRef<S, E> params = available_tasks_list.front();

      // int s0 = params.ntt_cpu->ntt_sub_logn.h0_layers_sub_logn[h1_layer][0];
      // int s1 = params.ntt_cpu->ntt_sub_logn.h0_layers_sub_logn[h1_layer][1];
      // int s2 = params.ntt_cpu->ntt_sub_logn.h0_layers_sub_logn[h1_layer][2];
      // int idx = params.reorder                ? 1<<(s0+s1) :
      //           params.task_c.h0_layer_idx==0 ? params.task_c.h0_subntt_idx + (params.task_c.h0_block_idx << s1) :
      //           params.task_c.h0_layer_idx==1 ? params.task_c.h0_subntt_idx + (params.task_c.h0_block_idx << s0) +
      //           (1<<(s1+s2)) :
      //         /*params.task_c.h0_layer_idx==2*/ params.task_c.h0_block_idx + (1<<(s1+s2)) + (1<<(s0+s2));

      // Assign the parameters to the available task
      available_task->set_ntt_cpu(params.ntt_cpu);
      available_task->set_input(params.input);
      available_task->set_coordinates(params.task_c);
      // available_task->set_index(idx);
      available_task->set_reorder(params.reorder);

      // Remove the task from the list
      available_tasks_list.erase(available_tasks_list.begin());
      return true;
    }
    return false;
  }

  // Function to set a task as completed and update dependencies
  template <typename S, typename E>
  eIcicleError NttTasksManagerRef<S, E>::set_task_as_completed(NttTaskRef<S, E>& completed_task, int nof_subntts_l2)
  {
    ntt_cpu::NttTaskCordinatesRef task_c = completed_task.get_coordinates();
    int nof_h0_layers = counters[task_c.h1_layer_idx].get_nof_h0_layers();
    // Update dependencies in counters
    if (counters[task_c.h1_layer_idx].decrement_counter(task_c)) {
      if (task_c.h0_layer_idx < nof_h0_layers - 1) {
        int nof_pointing_to_counter =
          (task_c.h0_layer_idx == nof_h0_layers - 1)
            ? 1
            : counters[task_c.h1_layer_idx].get_nof_pointing_to_counter(task_c.h0_layer_idx + 1);
        int stride = nof_subntts_l2 / nof_pointing_to_counter;
        for (int i = 0; i < nof_pointing_to_counter; i++) { // TODO - improve efficiency using make_move_iterator
          NttTaskCordinatesRef next_task_c =
            task_c.h0_layer_idx == 0
              ? NttTaskCordinatesRef{task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx + 1, task_c.h0_block_idx, i}
              /*task_c.h0_layer_idx==1*/
              : NttTaskCordinatesRef{
                  task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx + 1,
                  (task_c.h0_subntt_idx + stride * i), 0};
          if (waiting_tasks_map.find(next_task_c) != waiting_tasks_map.end()) {
            available_tasks_list.push_back(waiting_tasks_map[next_task_c]);
            waiting_tasks_map.erase(next_task_c);
          } else {
            ICICLE_LOG_ERROR << "Task not found in waiting_tasks_map: h0_layer_idx: " << next_task_c.h0_layer_idx
                             << ", h0_block_idx: " << next_task_c.h0_block_idx
                             << ", h0_subntt_idx: " << next_task_c.h0_subntt_idx;
          }
        }
      } else {
        // Reorder the output
        NttTaskCordinatesRef next_task_c = {task_c.h1_layer_idx, task_c.h1_subntt_idx, nof_h0_layers, 0, 0};

        if (waiting_tasks_map.find(next_task_c) != waiting_tasks_map.end()) {
          available_tasks_list.push_back(waiting_tasks_map[next_task_c]);
          waiting_tasks_map.erase(next_task_c);
        } else {
          ICICLE_LOG_ERROR << "Task not found in waiting_tasks_map";
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  //////////////////////////// NttCpuRef Implementation ////////////////////////////

  template <typename S, typename E>
  int NttCpuRef<S, E>::bit_reverse(int n, int logn)
  {
    int rev = 0;
    for (int j = 0; j < logn; ++j) {
      if (n & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  }

  template <typename S, typename E>
  // inline uint64_t NttCpuRef<S, E>::idx_in_mem(
  uint64_t NttCpuRef<S, E>::idx_in_mem(NttTaskCordinatesRef ntt_task_cordinates, int element)
  {
    int s0 = this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
    int s1 = this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
    int s2 = this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
    switch (ntt_task_cordinates.h0_layer_idx) {
    case 0:
      return ntt_task_cordinates.h0_block_idx + ((ntt_task_cordinates.h0_subntt_idx + (element << s1)) << s2);
    case 1:
      return ntt_task_cordinates.h0_block_idx + ((element + (ntt_task_cordinates.h0_subntt_idx << s1)) << s2);
    case 2:
      return ((ntt_task_cordinates.h0_block_idx << (s1 + s2)) & ((1 << (s0 + s1 + s2)) - 1)) +
             (((ntt_task_cordinates.h0_block_idx << (s1 + s2)) >> (s0 + s1 + s2)) << s2) + element;
    default:
      ICICLE_ASSERT(false) << "Unsupported layer";
    }
    return -1;
  }

  template <typename S, typename E>
  eIcicleError
  NttCpuRef<S, E>::reorder_by_bit_reverse(NttTaskCordinatesRef ntt_task_cordinates, E* elements, bool is_top_hirarchy)
  {
    uint64_t subntt_size =
      is_top_hirarchy ? (1 << (this->ntt_sub_logn.logn))
                      : 1 << this->ntt_sub_logn
                               .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int subntt_log_size =
      is_top_hirarchy
        ? (this->ntt_sub_logn.logn)
        : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * original_size;
      uint64_t rev;
      uint64_t i_mem_idx;
      uint64_t rev_mem_idx;
      for (uint64_t i = 0; i < subntt_size; ++i) {
        rev = this->bit_reverse(i, subntt_log_size);
        if (!is_top_hirarchy) {
          i_mem_idx = this->idx_in_mem(ntt_task_cordinates, i);
          rev_mem_idx = this->idx_in_mem(ntt_task_cordinates, rev);
        } else {
          i_mem_idx = i;
          rev_mem_idx = rev;
        }
        if (i < rev) {
          if (i_mem_idx < original_size && rev_mem_idx < original_size) { // Ensure indices are within bounds
            std::swap(current_elements[stride * i_mem_idx], current_elements[stride * rev_mem_idx]);
          } else {
            // Handle out-of-bounds error
            ICICLE_LOG_ERROR << "i=" << i << ", rev=" << rev << ", original_size=" << original_size;
            ICICLE_LOG_ERROR << "Index out of bounds: i_mem_idx=" << i_mem_idx << ", rev_mem_idx=" << rev_mem_idx;
            return eIcicleError::INVALID_ARGUMENT;
          }
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  void NttCpuRef<S, E>::dit_ntt(E* elements, NttTaskCordinatesRef ntt_task_cordinates, const S* twiddles) // R --> N
  {
    uint64_t subntt_size =
      1 << this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (1 << this->ntt_sub_logn.logn);
      for (int len = 2; len <= subntt_size; len <<= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (this->domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, i + j);
            uint64_t v_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, i + j + half_len);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx] * twiddles[tw_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = u - v;
          }
        }
      }
    }
  }

  template <typename S, typename E>
  void NttCpuRef<S, E>::dif_ntt(E* elements, NttTaskCordinatesRef ntt_task_cordinates, const S* twiddles)
  {
    uint64_t subntt_size =
      1 << this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (1 << this->ntt_sub_logn.logn);
      for (int len = subntt_size; len >= 2; len >>= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (this->domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, i + j);
            uint64_t v_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, i + j + half_len);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = (u - v) * twiddles[tw_idx];
          }
        }
      }
    }
  }

  template <typename S, typename E>
  eIcicleError NttCpuRef<S, E>::coset_mul(
    E* elements, const S* twiddles, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset)
  {
    uint64_t size = 1 << this->ntt_sub_logn.logn;
    uint64_t i_mem_idx;
    int idx;
    int batch_stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * size;
      if (arbitrary_coset) {
        for (int i = 1; i < size; ++i) {
          idx = this->config.columns_batch ? batch : i;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * arbitrary_coset[i];
        }
      } else if (coset_stride != 0) {
        for (int i = 1; i < size; ++i) {
          idx = coset_stride * i;
          idx = this->direction == NTTDir::kForward ? idx : this->domain_max_size - idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * twiddles[idx];
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  eIcicleError NttCpuRef<S, E>::reorder_input(E* input)
  { // TODO shanie future - consider using an algorithm for efficient reordering
    uint64_t size = 1 << this->ntt_sub_logn.logn;
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    auto temp_input = std::make_unique<E[]>(this->config.batch_size * size);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? input + batch : input + batch * size;
      E* current_temp_input = this->config.columns_batch ? temp_input.get() + batch : temp_input.get() + batch * size;
      uint64_t idx = 0;
      uint64_t new_idx = 0;
      int cur_ntt_log_size = this->ntt_sub_logn.h1_layers_sub_logn[0];
      int next_ntt_log_size = this->ntt_sub_logn.h1_layers_sub_logn[1];
      for (int i = 0; i < size; i++) {
        int subntt_idx = i >> cur_ntt_log_size;
        int element = i & ((1 << cur_ntt_log_size) - 1);
        new_idx = subntt_idx + (element << next_ntt_log_size);
        current_temp_input[stride * i] = current_elements[stride * new_idx];
      }
    }
    std::copy(temp_input.get(), temp_input.get() + this->config.batch_size * size, input);
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  void NttCpuRef<S, E>::refactor_and_reorder(E* elements, const S* twiddles)
  {
    int sntt_size = 1 << this->ntt_sub_logn.h1_layers_sub_logn[1];
    int nof_sntts = 1 << this->ntt_sub_logn.h1_layers_sub_logn[0];
    int ntt_size = 1 << (this->ntt_sub_logn.h1_layers_sub_logn[0] + this->ntt_sub_logn.h1_layers_sub_logn[1]);
    uint64_t temp_elements_size = ntt_size * this->config.batch_size;
    auto temp_elements =
      std::make_unique<E[]>(temp_elements_size); // TODO shanie - consider using an algorithm for sorting in-place
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* cur_layer_output = this->config.columns_batch ? elements + batch : elements + batch * ntt_size;
      E* cur_temp_elements =
        this->config.columns_batch ? temp_elements.get() + batch : temp_elements.get() + batch * ntt_size;
      for (int sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (int elem = 0; elem < sntt_size; elem++) {
          uint64_t tw_idx = (this->direction == NTTDir::kForward)
                              ? ((this->domain_max_size / ntt_size) * sntt_idx * elem)
                              : this->domain_max_size - ((this->domain_max_size / ntt_size) * sntt_idx * elem);
          cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
            cur_layer_output[stride * (elem * nof_sntts + sntt_idx)] * twiddles[tw_idx];
          // std::cout << "REF: h1_subntt_idx=\t" << elem << ",\tnew_idx=\t" << sntt_idx << ",\ttw_idx=\t" << tw_idx <<
          // std::endl; if (elem == 0){
          //   std::cout << "REF: h1_subntt_idx=\t" << elem << ",\telem=\t" << stride * (elem * nof_sntts + sntt_idx) <<
          //   ",\ttw_idx=\t" << tw_idx << ",\twiddles[tw_idx+elem]=\t" << twiddles[tw_idx+sntt_idx] << std::endl;
          // }
        }
      }
    }
    std::copy(temp_elements.get(), temp_elements.get() + temp_elements_size, elements);
  }

  template <typename S, typename E>
  eIcicleError
  NttCpuRef<S, E>::reorder_output(E* output, NttTaskCordinatesRef ntt_task_cordinates, bool is_top_hirarchy)
  { // TODO shanie future - consider using an algorithm for efficient reordering
    bool is_only_h0 = this->ntt_sub_logn.h1_layers_sub_logn[0] == 0;
    uint64_t size = (is_top_hirarchy || is_only_h0)
                      ? 1 << this->ntt_sub_logn.logn
                      : 1 << this->ntt_sub_logn.h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx];
    uint64_t temp_output_size = this->config.columns_batch ? size * this->config.batch_size : size;
    auto temp_output = std::make_unique<E[]>(temp_output_size);
    uint64_t idx = 0;
    uint64_t mem_idx = 0;
    uint64_t new_idx = 0;
    int subntt_idx;
    int element;
    int s0 = is_top_hirarchy ? this->ntt_sub_logn.h1_layers_sub_logn[0]
                             : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
    int s1 = is_top_hirarchy ? this->ntt_sub_logn.h1_layers_sub_logn[1]
                             : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
    int s2 = is_top_hirarchy ? this->ntt_sub_logn.h1_layers_sub_logn[2]
                             : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
    int p0, p1, p2;
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    int rep = this->config.columns_batch ? this->config.batch_size : 1;
    E* h1_subntt_output =
      output +
      stride * (ntt_task_cordinates.h1_subntt_idx
                << NttCpuRef<S, E>::ntt_sub_logn
                     .h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size //TODO
                                                                             // - NttCpuRef or this?
    for (int batch = 0; batch < rep; ++batch) {
      E* current_elements = this->config.columns_batch
                              ? h1_subntt_output + batch
                              : h1_subntt_output; // if columns_batch=false, then output is already shifted by
                                                  // batch*size when calling the function
      E* current_temp_output = this->config.columns_batch ? temp_output.get() + batch : temp_output.get();
      for (int i = 0; i < size; i++) {
        if (s2) {
          p0 = (i >> (s1 + s2));
          p1 = (((i >> s2) & ((1 << (s1)) - 1)) << s0);
          p2 = ((i & ((1 << s2) - 1)) << (s0 + s1));
          new_idx = p0 + p1 + p2;
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        } else {
          subntt_idx = i >> s1;
          element = i & ((1 << s1) - 1);
          new_idx = subntt_idx + (element << s0);
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
    }
    std::copy(temp_output.get(), temp_output.get() + temp_output_size, h1_subntt_output);
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  void NttCpuRef<S, E>::refactor_output_h0(E* elements, NttTaskCordinatesRef ntt_task_cordinates, const S* twiddles)
  {
    int h0_subntt_size = 1 << NttCpuRef<S, E>::ntt_sub_logn
                                .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int h0_nof_subntts = 1 << NttCpuRef<S, E>::ntt_sub_logn
                                .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]; // only relevant for layer 1
    int i, j, i_0;
    int ntt_size = ntt_task_cordinates.h0_layer_idx == 0
                     ? 1
                         << (NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
                             NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1])
                     : 1
                         << (NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
                             NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1] +
                             NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]);
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t original_size = (1 << NttCpuRef<S, E>::ntt_sub_logn.logn);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* h1_subntt_elements =
        elements +
        stride * (ntt_task_cordinates.h1_subntt_idx
                  << NttCpuRef<S, E>::ntt_sub_logn
                       .h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size
      E* elements_of_current_batch =
        this->config.columns_batch ? h1_subntt_elements + batch : h1_subntt_elements + batch * original_size;
      for (int elem = 0; elem < h0_subntt_size; elem++) {
        uint64_t elem_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, elem);
        i = (ntt_task_cordinates.h0_layer_idx == 0) ? elem : elem * h0_nof_subntts + ntt_task_cordinates.h0_subntt_idx;
        j = (ntt_task_cordinates.h0_layer_idx == 0) ? ntt_task_cordinates.h0_subntt_idx
                                                    : ntt_task_cordinates.h0_block_idx;
        uint64_t tw_idx = (this->direction == NTTDir::kForward)
                            ? ((this->domain_max_size / ntt_size) * j * i)
                            : this->domain_max_size - ((this->domain_max_size / ntt_size) * j * i);
        // if (ntt_task_cordinates.h0_layer_idx == 1){
        //   ICICLE_LOG_DEBUG << "elem_mem_idx: " << elem_mem_idx << ", i: " << i << ", j: " << j << ", tw_idx: " <<
        //   tw_idx;
        // }
        elements_of_current_batch[elem_mem_idx] = elements_of_current_batch[elem_mem_idx] * twiddles[tw_idx];
        // ICICLE_LOG_INFO << "twiddles[" << tw_idx << "]: " << twiddles[tw_idx];
      }
    }
  }

  template <typename S, typename E>
  eIcicleError NttCpuRef<S, E>::h1_cpu_ntt(
    E* input, NttTaskCordinatesRef ntt_task_cordinates, NttTasksManagerRef<S, E>& ntt_tasks_manager)
  {
    // ICICLE_LOG_INFO <<"NttCpuRef<S, E>::h1_cpu_ntt";
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    int nof_h0_layers = (NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2] != 0) ? 3
                        : (NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1] != 0)
                          ? 2
                          : 1;
    // Assuming that NTT fits in the cache, so we split the NTT to layers and calculate them one after the other.
    // Subntts inside the same laye are calculate in parallel.
    // Sorting is not needed, since the elements needed for each subntt are close to each other in memory.
    // Instead of sorting, we are using the function idx_in_mem to calculate the memory index of each element.
    for (ntt_task_cordinates.h0_layer_idx = 0;
         ntt_task_cordinates.h0_layer_idx <
         NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx].size();
         ntt_task_cordinates.h0_layer_idx++) {
      if (ntt_task_cordinates.h0_layer_idx == 0) {
        // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx;
        int log_nof_subntts = NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
        int log_nof_blocks = NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks);
             ntt_task_cordinates.h0_block_idx++) {
          for (ntt_task_cordinates.h0_subntt_idx = 0; ntt_task_cordinates.h0_subntt_idx < (1 << log_nof_subntts);
               ntt_task_cordinates.h0_subntt_idx++) {
            // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx:
            // " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
            ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
          }
        }
      }
      if (
        ntt_task_cordinates.h0_layer_idx == 1 &&
        NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]) {
        // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx;
        int log_nof_subntts = NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
        int log_nof_blocks = NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks);
             ntt_task_cordinates.h0_block_idx++) {
          for (ntt_task_cordinates.h0_subntt_idx = 0; ntt_task_cordinates.h0_subntt_idx < (1 << log_nof_subntts);
               ntt_task_cordinates.h0_subntt_idx++) {
            ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
          }
        }
      }
      if (
        ntt_task_cordinates.h0_layer_idx == 2 &&
        NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]) {
        ntt_task_cordinates.h0_subntt_idx = 0; // not relevant for layer 2
        // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx;
        int log_nof_blocks = NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
                             NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks);
             ntt_task_cordinates.h0_block_idx++) {
          // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: "
          // << ntt_task_cordinates.h0_block_idx;
          ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
        }
      }
    }
    // Sort the output at the end so that elements will be in right order.
    // TODO SHANIE  - After implementing for different ordering, maybe this should be in a different place
    //              - When implementing real parallelism, consider sorting in parallel and in-place
    int nof_h0_subntts =
      (nof_h0_layers == 1)
        ? (1 << NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1])
      : (nof_h0_layers == 2)
        ? (1 << NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0])
        : 1;
    int nof_h0_blocks =
      (nof_h0_layers != 3)
        ? (1 << NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2])
        : (1
           << (NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
               NttCpuRef<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]));
    if (nof_h0_layers > 1) {
      // ICICLE_LOG_DEBUG << "h1_cpu_ntt: PUSH REORDER TASK h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ",
      // h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " <<
      // ntt_task_cordinates.h0_subntt_idx;
      ntt_task_cordinates = {ntt_task_cordinates.h1_layer_idx, ntt_task_cordinates.h1_subntt_idx, nof_h0_layers, 0, 0};
      ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, true); // reorder=true
    }
    // if (nof_h0_layers>1) { // at least 2 layers
    //   if (this->config.columns_batch) {
    //     this->reorder_output(input, ntt_task_cordinates, false);
    //   } else {
    //     for (int b = 0; b < this->config.batch_size; b++) {
    //       this->reorder_output(input + b * original_size, ntt_task_cordinates, false);
    //     }
    //   }
    // }
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  eIcicleError NttCpuRef<S, E>::h0_cpu_ntt(E* input, NttTaskCordinatesRef ntt_task_cordinates)
  {
    // ICICLE_LOG_INFO << "NttCpuRef<S, E>::h0_cpu_ntt";
    const uint64_t subntt_size =
      (1 << NttCpuRef<S, E>::ntt_sub_logn
              .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx]);
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    const uint64_t total_memory_size = original_size * this->config.batch_size;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    int offset = this->config.columns_batch ? this->config.batch_size : 1;
    E* current_input =
      input + offset * (ntt_task_cordinates.h1_subntt_idx
                        << NttCpuRef<S, E>::ntt_sub_logn
                             .h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size

    this->reorder_by_bit_reverse(
      ntt_task_cordinates, current_input,
      false); // TODO - check if access the fixed indexes instead of reordering may be more efficient?

    // NTT/INTT
    this->dit_ntt(current_input, ntt_task_cordinates, twiddles); // R --> N

    if (
      ntt_task_cordinates.h0_layer_idx != 2 &&
      this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx + 1] !=
        0) {
      this->refactor_output_h0(input, ntt_task_cordinates, twiddles);
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  eIcicleError NttCpuRef<S, E>::handle_pushed_tasks(
    TasksManager<NttTaskRef<S, E>>* tasks_manager, NttTasksManagerRef<S, E>& ntt_tasks_manager, int h1_layer_idx)
  {
    // // Initialize the accumulator for get_available_task_to_run
    // double total_time_get_available_task_to_run = 0.0;
    // int call_count_get_available_task_to_run = 0;
    // // Initialize the accumulator for set_task_as_completed
    // double total_time_set_task_as_completed = 0.0;
    // int call_count_set_task_as_completed = 0;
    // int clean_completed_counter = 0;
    //       auto start_set_task_as_completed = std::chrono::high_resolution_clock::now();
    //       auto end_set_task_as_completed = std::chrono::high_resolution_clock::now();
    //       total_time_set_task_as_completed += std::chrono::duration<double, std::milli>(end_set_task_as_completed -
    //       start_set_task_as_completed).count();

    // ICICLE_LOG_INFO << "NttCpuRef<S, E>::handle_pushed_tasks";
    NttTaskRef<S, E>* task_slot = nullptr;
    NttTaskParamsRef<S, E> params;

    int nof_subntts_l2 = 1
                         << ((this->ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]) +
                             (this->ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1]));
    while (ntt_tasks_manager.tasks_to_do()) {
      // There are tasks that are available or waiting

      if (ntt_tasks_manager.available_tasks()) {
        // Task is available to dispatch
        task_slot = tasks_manager->get_idle_or_completed_task();
        if (task_slot->is_completed()) { ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l2); }
        params = ntt_tasks_manager.get_available_task();
        task_slot->set_params(params);
        ntt_tasks_manager.erase_task_from_available_tasks_list();
        task_slot->dispatch();
      } else {
        // Wait for available tasks
        task_slot = tasks_manager->get_completed_task();
        ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l2);
        if (ntt_tasks_manager.available_tasks()) {
          params = ntt_tasks_manager.get_available_task();
          task_slot->set_params(params);
          ntt_tasks_manager.erase_task_from_available_tasks_list();
          task_slot->dispatch();
        } else {
          task_slot->set_idle();
        }
      }
    }
    task_slot = tasks_manager->get_completed_task(); // Get the last task (reorder task)
    // ICICLE_LOG_INFO << std::fixed << std::setprecision(3)
    //                  << "Total time spent on get_available_task_to_run: " << total_time_get_available_task_to_run <<
    //                  " ms";
    // ICICLE_LOG_INFO << "call_count_get_available_task_to_run=" << call_count_get_available_task_to_run;
    // ICICLE_LOG_INFO << std::fixed << std::setprecision(3)
    //                  << "Total time spent on set_task_as_completed: " << total_time_set_task_as_completed << " ms";
    // ICICLE_LOG_INFO << "call_count_set_task_as_completed=" << call_count_set_task_as_completed;
    // ICICLE_LOG_INFO << "clean_completed_counter=" << clean_completed_counter;
    return eIcicleError::SUCCESS;
  }

} // namespace ntt_cpu