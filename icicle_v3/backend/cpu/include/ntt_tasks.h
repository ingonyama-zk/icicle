#pragma once
#include "icicle/backend/ntt_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/fields/field_config.h"
#include "icicle/vec_ops.h"
#include "tasks_manager.h"


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

  constexpr uint32_t layers_sub_logn[31][3] = {
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

  struct NttTaskCordinates {
    int h1_layer_idx=0;
    int h1_subntt_idx=0;
    int h0_layer_idx=0;
    int h0_subntt_idx=0;
    int h0_block_idx=0;

    // Comparison operators for map
    bool operator<(const NttTaskCordinates& other) const {
      return std::tie(h1_layer_idx, h1_subntt_idx, h0_layer_idx, h0_subntt_idx, h0_block_idx) <
        std::tie(other.h1_layer_idx, other.h1_subntt_idx, other.h0_layer_idx, other.h0_subntt_idx, other.h0_block_idx);
    }
  };
  
  struct NttSubLogn {
    int logn; // Original size of the problem
    std::vector<std::vector<int>> h0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<int> h1_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubLogn(int logn) : logn(logn)
    {
      if (logn > 15){
        // Initialize h1_layers_sub_logn
        h1_layers_sub_logn = std::vector<int>(
          std::begin(layers_sub_logn[logn]), 
          std::end(layers_sub_logn[logn])
        );
        // Initialize h0_layers_sub_logn
        h0_layers_sub_logn = {std::vector<int>(
          std::begin(layers_sub_logn[h1_layers_sub_logn[0]]), 
          std::end(layers_sub_logn[h1_layers_sub_logn[0]])
        ), 
        std::vector<int>(
          std::begin(layers_sub_logn[h1_layers_sub_logn[1]]), 
          std::end(layers_sub_logn[h1_layers_sub_logn[1]])
        )};
      } else {
        h1_layers_sub_logn = {0, 0, 0};
        h0_layers_sub_logn = {std::vector<int>(
          std::begin(layers_sub_logn[logn]), 
          std::end(layers_sub_logn[logn])
        ), {0, 0, 0}};
      }
      ICICLE_LOG_DEBUG << "NttTaskInfo: h1_layers_sub_logn: " << h1_layers_sub_logn[0] << ", " << h1_layers_sub_logn[1] << ", " << h1_layers_sub_logn[2];
      ICICLE_LOG_DEBUG << "NttTaskInfo: h0_layers_sub_logn[0]: " << h0_layers_sub_logn[0][0] << ", " << h0_layers_sub_logn[0][1] << ", " << h0_layers_sub_logn[0][2];
      ICICLE_LOG_DEBUG << "NttTaskInfo: h0_layers_sub_logn[1]: " << h0_layers_sub_logn[1][0] << ", " << h0_layers_sub_logn[1][1] << ", " << h0_layers_sub_logn[1][2];
    }
  };

  template <typename S = scalar_t, typename E = scalar_t>
  class NttCpu{
    public:
      NttSubLogn ntt_sub_logn;
      NttTaskCordinates ntt_task_cordinates;
      NTTDir direction;
      const NTTConfig<S>& config;
      int domain_max_size;
      const S* twiddles;

      NttCpu(int logn, NTTDir direction, const NTTConfig<S>& config, int domain_max_size, const S* twiddles) :  ntt_sub_logn(logn), direction(direction), config(config), domain_max_size(domain_max_size), twiddles(twiddles) {}

      int bit_reverse(int n, int logn);
      uint64_t idx_in_mem(NttTaskCordinates ntt_task_cordinates, int element);
      eIcicleError reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy);
      void dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles);
      void dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles);
      eIcicleError coset_mul(E* elements, const S* twiddles, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset);
      eIcicleError reorder_input(E* input);
      void refactor_and_reorder(E* elements, const S* twiddles);
      eIcicleError reorder_output(E* elements, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy);
      void refactor_output_h0(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles);
      eIcicleError h1_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates);
      eIcicleError h0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates);
  };


  class TasksDependenciesCounters {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx);

    // Function to get a counter for a given task
    std::shared_ptr<int> get_counter(const NttTaskCordinates& task_c);


    // Function to decrement the counter for a given task and check if it is ready to execute
    void decrement_counter(NttTaskCordinates ntt_task_cordinates);
    
    // Other member functions as needed...

  private:

    int h1_layer_idx;
    
    // Each h1_subntt has its own set of counters
    std::vector<std::vector<std::vector<std::shared_ptr<int>>>> h0_counters;  // [h1_subntt_idx][layer_idx][counter_idx]

    // One counter for each h1_subntt to signal the end of the h1_subntt. each h0_subntt of layer 2 will decrement this counter when it finishes
    std::vector<std::shared_ptr<int>> h1_counters;  // [h1_subntt_idx]

  };

  TasksDependenciesCounters::TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx)
    : h0_counters(ntt_sub_logn.h1_layers_sub_logn[1-h1_layer_idx]), //nof_h1_subntts = h1_layers_sub_logn[1-h1_layer_idx]. 
      h1_counters(ntt_sub_logn.h1_layers_sub_logn[1-h1_layer_idx]) { // Initialize h1_counters with N0 * N1

    int nof_h0_layers = ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2] ? 3 : (ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1] ? 2 : 1);
      
    for (int h1_subntt_idx = 0; h1_subntt_idx < ntt_sub_logn.h1_layers_sub_logn[1-h1_layer_idx]; ++h1_subntt_idx) {
      h0_counters[h1_subntt_idx].resize(3); // Assuming 3 layers (0, 1, 2)
      
      // Initialize counters for layer 0 - 1 counter1 initialized with 0.
      h0_counters[h1_subntt_idx][0].resize(1);
      h0_counters[h1_subntt_idx][0][0] = std::make_shared<int>(0);

      // Initialize counters for layer 1 - N2 counters initialized with N1. if nof_h0_layers <=2, layer 1 is skipped
      if (nof_h0_layers > 2) {
        h0_counters[h1_subntt_idx][1].resize(ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2]);
        for (int counter_idx = 0; counter_idx < ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2]; ++counter_idx) {
          h0_counters[h1_subntt_idx][1][counter_idx] = std::make_shared<int>(ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1]);
        }
      }

      // Initialize counters for layer 2 - N0 counters initialized with N2. if nof_h0_layers <=1, also layer 2 is skipped
      if (nof_h0_layers > 1) {
        h0_counters[h1_subntt_idx][2].resize(ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]);
        for (int counter_idx = 0; counter_idx < ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]; ++counter_idx) {
          h0_counters[h1_subntt_idx][2][counter_idx] = std::make_shared<int>(ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2]);
        }
      }
    
      // Initialize h1_counters with N0 * N1
      h1_counters[h1_subntt_idx] = std::make_shared<int>(ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0] * ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1]);
    
    }
  }

  std::shared_ptr<int> TasksDependenciesCounters::get_counter(const NttTaskCordinates& task_c) {
      if (task_c.h0_layer_idx == 1) {
        return h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][task_c.h0_block_idx];
      } else if (task_c.h0_layer_idx == 2) {
        return h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][task_c.h0_subntt_idx];
      } else {
        // Handle other cases or throw an exception
        return nullptr; // Default or error value
      }
  }

  void TasksDependenciesCounters::decrement_counter(NttTaskCordinates ntt_task_cordinates) {

    if (ntt_task_cordinates.h0_layer_idx < 2) {
      // Extract the coordinates from the task
      int h1_subntt_idx = ntt_task_cordinates.h1_subntt_idx;
      int counter_h0_layer_idx = ntt_task_cordinates.h0_layer_idx + 1;
      int counter_group_idx = counter_h0_layer_idx==1 ? ntt_task_cordinates.h0_block_idx :
                            /*counter_h0_layer_idx==2*/ ntt_task_cordinates.h0_subntt_idx;

      // Decrement the counter for the given h1_subntt_idx, h0_block_idx, h0_layer_idx
        std::shared_ptr<int>& counter_ptr = h0_counters[h1_subntt_idx][counter_h0_layer_idx][counter_group_idx];
        (*counter_ptr)--;

      if (*counter_ptr == 0) {
        // TODO - Handle the logic for tasks that are now ready to execute
      }
    } else {
      // Decrement the counter for the given h1_subntt_idx
      std::shared_ptr<int>& h1_counter_ptr = h1_counters[ntt_task_cordinates.h1_subntt_idx];
      (*h1_counter_ptr)--;

      if (*h1_counter_ptr == 0) {
        // TODO - Handle the logic for tasks that are now ready to execute
      }
    }
  }

  struct NttTaskStatus {
    std::shared_ptr<int> counter; // Number of tasks that must be completed before this task can run
    bool done = false;         // True if the task has been completed
    bool reorder = false;      // Whether the task is to reorder
  };

  using NttTasksStatus = std::map<NttTaskCordinates, NttTaskStatus>;


  template<typename S = scalar_t, typename E = scalar_t>
  class NttTask : public TaskBase {
  public:
    NttTask(
      NttCpu<S, E>& ntt_cpu,   // Reference to an NttCpu instance
      E* input,
      NttTaskCordinates ntt_task_cordinates, 
      bool reorder = false)
      : ntt_cpu(ntt_cpu), input(input), ntt_task_cordinates(ntt_task_cordinates), reorder(reorder) {}

    void execute() override {
      if (reorder) {
        // Reorder the input if necessary before performing h0_cpu_ntt
        ntt_cpu.reorder_by_bit_reverse(ntt_task_cordinates, input, false);
      } else {
      // Execute the h0_cpu_ntt using the provided NttCpu instance
        ntt_cpu.h0_cpu_ntt(input, ntt_task_cordinates);
      }
    }

    NttTaskCordinates get_coordinates() const {
      return ntt_task_cordinates;
    }

  private:
    NttCpu<S, E>& ntt_cpu;  // Reference to NttCpu instance
    E* input;
    NttTaskCordinates ntt_task_cordinates;
    bool reorder;
  };


  template<typename S = scalar_t, typename E = scalar_t>
  class NttTasksManager {
  public:
      NttTasksManager(int logn) 
          : tasks_status(logn > 15 ? 2 : 1), 
            counters(logn > 15 ? 2 : 1, TasksDependenciesCounters(NttSubLogn(logn), 0)) {
          if (logn > 15) {
              counters[1] = TasksDependenciesCounters(NttSubLogn(logn), 1);
          }
      }
      // Function to add a new task to the manager
      bool push_task(NttCpu<S, E>& ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder);

      // Function to get an available task to run
      bool get_available_task_to_run(NttTask<S, E>* available_task, int h1_layer);

      // Function to set a task as completed and update dependencies
      eIcicleError set_task_as_completed(NttTask<S, E>& completed_task);

  private:
      // Function to get the counter for a specific task based on its coordinates
      std::shared_ptr<int> get_counter_for_task(const NttTaskCordinates& task_c) {
        return counters[task_c.h1_layer_idx].get_counter(task_c);
      }

      std::vector<NttTasksStatus> tasks_status;  // Status of tasks by layer
      std::vector<TasksDependenciesCounters> counters;  // Dependencies counters by layer
      std::vector<NttTask<S, E>> task_list; // List of all tasks

  };

  template<typename S, typename E>
  bool NttTasksManager<S, E>::push_task(NttCpu<S, E>& ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder) {
    // Create a shared pointer to the relevant counter
    auto counter = get_counter_for_task(task_c);

    if (tasks_status[task_c.h1_layer_idx].find(task_c) == tasks_status[task_c.h1_layer_idx].end()) {
      NttTaskStatus status = {counter, false, reorder};
      tasks_status[task_c.h1_layer_idx][task_c] = status;

      NttTask<S, E> new_task(ntt_cpu, input, task_c, reorder);
      task_list.push_back(new_task);

      return true;
    }
    return false;
  }

  template<typename S, typename E>
  bool NttTasksManager<S, E>::get_available_task_to_run(NttTask<S, E>* available_task, int h1_layer) {

    for (NttTask<S, E>& task : task_list) {
      NttTaskStatus& status = tasks_status[h1_layer][task.get_coordinates()];
      if (status.counter && *status.counter == 0 && !status.done) {
        *available_task = &task;
        status.counter = nullptr;
        return true;
      }
    }
    return false;
  }

  // Function to set a task as completed and update dependencies
  template<typename S, typename E>
  eIcicleError NttTasksManager<S, E>::set_task_as_completed(NttTask<S, E>& completed_task) {
    auto& status = tasks_status[completed_task.get_coordinates().h1_layer_idx][completed_task.get_coordinates()];
    status.done = true;
    // Update dependencies in counters
    counters[completed_task.get_coordinates().h1_layer_idx].decrement_counter(completed_task.get_coordinates());
    return eIcicleError::SUCCESS;
  }

} // namespace ntt_cpu