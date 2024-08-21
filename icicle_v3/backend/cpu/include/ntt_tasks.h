#pragma once
#include "icicle/backend/ntt_backend.h"
#include "icicle/errors.h"
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

  constexpr uint32_t layers_sub_logn[31][3] = {
    // {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {1, 1, 1},   {2, 1, 1},   {2, 2, 1},   {3, 2, 1},   {4, 3, 0},
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

  struct NttTaskCordinates {
    int h1_layer_idx=0;
    int h1_subntt_idx=0;
    int h0_layer_idx=0;
    int h0_block_idx=0;
    int h0_subntt_idx=0;

    // Comparison operators for map
    bool operator<(const NttTaskCordinates& other) const {
      return std::tie(h1_layer_idx, h1_subntt_idx, h0_layer_idx, h0_block_idx, h0_subntt_idx) <
        std::tie(other.h1_layer_idx, other.h1_subntt_idx, other.h0_layer_idx, other.h0_block_idx, other.h0_subntt_idx);
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
      // ICICLE_LOG_DEBUG << "NttTaskInfo: h1_layers_sub_logn: " << h1_layers_sub_logn[0] << ", " << h1_layers_sub_logn[1] << ", " << h1_layers_sub_logn[2];
      // ICICLE_LOG_DEBUG << "NttTaskInfo: h0_layers_sub_logn[0]: " << h0_layers_sub_logn[0][0] << ", " << h0_layers_sub_logn[0][1] << ", " << h0_layers_sub_logn[0][2];
      // ICICLE_LOG_DEBUG << "NttTaskInfo: h0_layers_sub_logn[1]: " << h0_layers_sub_logn[1][0] << ", " << h0_layers_sub_logn[1][1] << ", " << h0_layers_sub_logn[1][2];
    }
  };

  template<typename S, typename E>
  class NttTask;

  template<typename S, typename E>
  class NttTasksManager;

  template <typename S = scalar_t, typename E = scalar_t>
  class NttCpu{
    public:
      NttSubLogn ntt_sub_logn;
      NTTDir direction;
      const NTTConfig<S>& config;
      int domain_max_size;
      const S* twiddles;

      NttCpu(int logn, NTTDir direction, const NTTConfig<S>& config, int domain_max_size, const S* twiddles) :  ntt_sub_logn(logn), direction(direction), config(config), domain_max_size(domain_max_size), twiddles(twiddles) {
        // ICICLE_LOG_DEBUG << "NttCpu constructor";
      }

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
      eIcicleError h1_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates, NttTasksManager<S, E>& ntt_tasks_manager);
      eIcicleError h0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates);
      eIcicleError handle_pushed_tasks(TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, int h1_layer_idx);
  };


  class TasksDependenciesCounters {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx);

    // Function to get a counter for a given task
    std::shared_ptr<int> get_counter(const NttTaskCordinates& task_c, bool reorder);

    // Function to decrement the counter for a given task and check if it is ready to execute. if so, return true
    bool decrement_counter(NttTaskCordinates ntt_task_cordinates);
    int get_nof_pointing_to_counter(int h0_layer_idx) { return nof_pointing_to_counter[h0_layer_idx]; }
    int get_nof_h0_layers() { return nof_h0_layers; }
    
  private:

    int h1_layer_idx;
    int nof_h0_layers;
    std::vector<int> nof_pointing_to_counter; // Number of counters for each layer
    
    // Each h1_subntt has its own set of counters
    std::vector<std::vector<std::vector<std::shared_ptr<int>>>> h0_counters;  // [h1_subntt_idx][h0_layer_idx][h0_counter_idx]

    // One counter for each h1_subntt to signal the end of the h1_subntt. each h0_subntt of last h0_layer will decrement this counter when it finishes
    // and when it reaches 0, the h1_subntt is ready to reorder
    std::vector<std::shared_ptr<int>> h1_counters;  // [h1_subntt_idx]
  };

  struct NttTaskStatus {
    bool done = false;         // True if the task has been completed
    bool reorder = false;      // Whether the task is to reorder
  };

  template<typename S = scalar_t, typename E = scalar_t>
  struct NttTaskParams {
    NttCpu<S, E>* ntt_cpu;
    E* input;
    NttTaskCordinates task_c;
    bool reorder;
  };

  using NttTasksStatus = std::map<NttTaskCordinates, NttTaskStatus>;

  template<typename S = scalar_t, typename E = scalar_t>
  class NttTask : public TaskBase {
  public:
    // Default constructor
    NttTask() : ntt_cpu(nullptr), input(nullptr), reorder(false) {}

    // // Constructor with parameters
    // NttTask(
    //   NttCpu<S, E>& ntt_cpu,   // Reference to an NttCpu instance
    //   E* input,
    //   NttTaskCordinates ntt_task_cordinates, 
    //   bool reorder = false)
    //   : ntt_cpu(ntt_cpu), input(input), ntt_task_cordinates(ntt_task_cordinates), reorder(reorder) {}

    void execute() {
      auto thread_id = std::this_thread::get_id();  // Get the thread id
      if (reorder) {
        // if all h0_subntts are done, reorder the output
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing reorder start: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
        if (ntt_cpu->config.columns_batch) {
          ntt_cpu->reorder_output(input, ntt_task_cordinates, false);
        } else {
          for (int b = 0; b < ntt_cpu->config.batch_size; b++) {
            ntt_cpu->reorder_output(input + b * (1<<(ntt_cpu->ntt_sub_logn.logn)), ntt_task_cordinates, false);
          }
        }
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing reorder done:  h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
      } else {
      // Execute the h0_cpu_ntt using the provided NttCpu instance
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing ntt start: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
        ntt_cpu->h0_cpu_ntt(input, ntt_task_cordinates);
        // ICICLE_LOG_DEBUG << "Thread " << thread_id << ": Executing ntt done:  h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
        // for (int i = 0; i < 1<<ntt_cpu->ntt_sub_logn.logn; i++) {
        //   ICICLE_LOG_DEBUG << "Thread " << thread_id << ": input[" << i << "]: " << input[i];
        // }
      }
    }

    NttTaskCordinates get_coordinates() const {
      return ntt_task_cordinates;
    }

    bool is_reorder() const {
      return reorder;
    }
    void set_ntt_cpu(NttCpu<S, E>* cpu) { ntt_cpu = cpu; }
    void set_input(E* inp) { input = inp; }
    void set_coordinates(const NttTaskCordinates& coordinates) { ntt_task_cordinates = coordinates; }
    void set_reorder(bool reorder_val) { reorder = reorder_val; }

  private:
    NttCpu<S, E>* ntt_cpu;  // Reference to NttCpu instance
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
      nof_available_tasks = 0;
      nof_waiting_tasks = 0;
      // ICICLE_LOG_DEBUG << "NttTasksManager constructor";
    }
    // Add a new task to the ntt_task_manager
    eIcicleError push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder);

    // Function to get an available task to run
    bool get_available_task_to_run(NttTask<S, E>* available_task, int h1_layer); //if no availble task- available_task=null. if not waiting to be ready - return false

    // Set a task as completed and update dependencies
    eIcicleError set_task_as_completed(NttTask<S, E>& completed_task, int nof_subntts_l2);

    int nof_available_tasks;
    int nof_waiting_tasks;

  private:
    // Function to get the counter for a specific task based on its coordinates
    std::shared_ptr<int> get_counter_p_for_task(const NttTaskCordinates& task_c, bool reorder) {
      return counters[task_c.h1_layer_idx].get_counter(task_c, reorder);
    }

    std::vector<NttTasksStatus> tasks_status;  // Status of tasks by layer
    std::vector<TasksDependenciesCounters> counters;  // Dependencies counters by layer
    std::vector<NttTaskParams<S, E>> available_tasks_params_list;
    // std::vector<NttTaskParams<S, E>> waiting_tasks_params_list;
    std::map<NttTaskCordinates, NttTaskParams<S, E>> waiting_tasks_params_map;

  };

  //////////////////////////// NttTasksManager Implementation ////////////////////////////

  TasksDependenciesCounters::TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx)
    : h0_counters(1<<ntt_sub_logn.h1_layers_sub_logn[1-h1_layer_idx]), //nof_h1_subntts = h1_layers_sub_logn[1-h1_layer_idx]. 
      h1_counters(1<<ntt_sub_logn.h1_layers_sub_logn[1-h1_layer_idx]) { // Initialize h1_counters with N0 * N1

    nof_h0_layers = ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2] ? 3 : (ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1] ? 2 : 1);
    // ICICLE_LOG_DEBUG << "TasksDependenciesCounters: nof_h0_layers: " << nof_h0_layers;
    nof_pointing_to_counter.resize(nof_h0_layers);
    nof_pointing_to_counter[0] = 1;
    int l1_counter_size;
    int l2_counter_size;
    int l1_nof_counters;
    int l2_nof_counters;
    if (nof_h0_layers > 1) {
      // Initialize counters for layer 1 - N2 counters initialized with N1. 
      nof_pointing_to_counter[1] = 1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0];
      l1_nof_counters = 1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2];
      l1_counter_size = 1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1];
    }
    if (nof_h0_layers > 2) {
      // Initialize counters for layer 2 - N0 counters initialized with N2.
      nof_pointing_to_counter[2] = 1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1];
      l2_nof_counters = 1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0];
      l2_counter_size = 1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2];
    }

    
    for (int h1_subntt_idx = 0; h1_subntt_idx < (1<<ntt_sub_logn.h1_layers_sub_logn[1-h1_layer_idx]); ++h1_subntt_idx) {
      h0_counters[h1_subntt_idx].resize(3); // Assuming 3 layers (0, 1, 2)
      
      // Initialize counters for layer 0 - 1 counter1 initialized with 0.
      h0_counters[h1_subntt_idx][0].resize(1);
      h0_counters[h1_subntt_idx][0][0] = std::make_shared<int>(0); //[h1_subntt_idx][h0_layer_idx][h0_counter_idx]
      // ICICLE_LOG_DEBUG << "TasksDependenciesCounters: h0_counters["<<h1_subntt_idx<<"][0][0]: " << *h0_counters[h1_subntt_idx][0][0];

      if (nof_h0_layers > 1) {
        // Initialize counters for layer 1 - N2 counters initialized with N1. 
        h0_counters[h1_subntt_idx][1].resize(l1_nof_counters);
        for (int counter_idx = 0; counter_idx < l1_nof_counters; ++counter_idx) {
          h0_counters[h1_subntt_idx][1][counter_idx] = std::make_shared<int>(l1_counter_size);
          // ICICLE_LOG_DEBUG << "TasksDependenciesCounters: h0_counters["<<h1_subntt_idx<<"][1]["<<counter_idx<<"]: " << *h0_counters[h1_subntt_idx][1][counter_idx];
        }
      }
      if (nof_h0_layers > 2) {
        // Initialize counters for layer 2 - N0 counters initialized with N2.
        h0_counters[h1_subntt_idx][2].resize(l2_nof_counters);
        for (int counter_idx = 0; counter_idx < l2_nof_counters; ++counter_idx) {
          h0_counters[h1_subntt_idx][2][counter_idx] = std::make_shared<int>(l2_counter_size);
          // ICICLE_LOG_DEBUG << "TasksDependenciesCounters: h0_counters["<<h1_subntt_idx<<"][2]["<<counter_idx<<"]: " << *h0_counters[h1_subntt_idx][2][counter_idx];
        }
      }
    
      // Initialize h1_counters with N0 * N1
      int h1_counter_size = nof_h0_layers == 3 ? (1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]) * (1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1])
                          : nof_h0_layers == 2 ? (1<<ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]) : 0 ;
      h1_counters[h1_subntt_idx] = std::make_shared<int>(h1_counter_size);
      // ICICLE_LOG_DEBUG << "TasksDependenciesCounters: h1_counters["<<h1_subntt_idx<<"]: " << *h1_counters[h1_subntt_idx];
    
    }
  }

  std::shared_ptr<int> TasksDependenciesCounters::get_counter(const NttTaskCordinates& task_c, bool reorder) {
    if (reorder) {
      // ICICLE_LOG_DEBUG << "get_counter: h1_counters["<<task_c.h1_subntt_idx<<"]: " << *h1_counters[task_c.h1_subntt_idx];
      return h1_counters[task_c.h1_subntt_idx];
    }
    if (task_c.h0_layer_idx == 0) {
      // ICICLE_LOG_DEBUG << "get_counter: h0_counters["<<task_c.h1_subntt_idx<<"]["<<task_c.h0_layer_idx<<"][0]: " << *h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][0];
      return h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][0];
    }
    if (task_c.h0_layer_idx == 1) {
      // ICICLE_LOG_DEBUG << "get_counter: h0_counters["<<task_c.h1_subntt_idx<<"]["<<task_c.h0_layer_idx<<"]["<<task_c.h0_block_idx<<"]: " << *h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][task_c.h0_block_idx];
      return h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][task_c.h0_block_idx];
    }
    if (task_c.h0_layer_idx == 2) {
      // ICICLE_LOG_DEBUG << "get_counter: h0_counters["<<task_c.h1_subntt_idx<<"]["<<task_c.h0_layer_idx<<"]["<<task_c.h0_subntt_idx<<"]: " << *h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][task_c.h0_subntt_idx];
      return h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx][task_c.h0_block_idx/this->nof_pointing_to_counter[task_c.h0_layer_idx]];
    } else {
      ICICLE_LOG_ERROR << "get_counter: return nullptr";
      // Handle other cases or throw an exception
      return nullptr; // Default or error value
    }
    
  }

  bool TasksDependenciesCounters::decrement_counter(NttTaskCordinates task_c) {
    if (nof_h0_layers==1){
      return false;
    }
    if (task_c.h0_layer_idx < nof_h0_layers-1) {
      // Extract the coordinates from the task
      int counter_group_idx = task_c.h0_layer_idx==0 ? task_c.h0_block_idx :
                            /*task_c.h0_layer_idx==1*/ task_c.h0_subntt_idx;

      std::shared_ptr<int>& counter_ptr = h0_counters[task_c.h1_subntt_idx][task_c.h0_layer_idx + 1][counter_group_idx];
      (*counter_ptr)--;

      if (*counter_ptr == 0) {
        return true;
      }
    } else {
      // Decrement the counter for the given h1_subntt_idx
      std::shared_ptr<int>& h1_counter_ptr = h1_counters[task_c.h1_subntt_idx];
      (*h1_counter_ptr)--;

      if (*h1_counter_ptr == 0) {
        return true;
      }
    }
    return false;
  }

  
  //////////////////////////// NttTasksManager Implementation ////////////////////////////
  
  template<typename S, typename E>
  eIcicleError NttTasksManager<S, E>::push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder) {

    if (tasks_status[task_c.h1_layer_idx].find(task_c) == tasks_status[task_c.h1_layer_idx].end()) {
      NttTaskStatus status = {false, reorder};
      tasks_status[task_c.h1_layer_idx][task_c] = status;
        
      // Create a new NttTaskParams and add it to the available_tasks_params_list
      NttTaskParams<S, E> params = {ntt_cpu, input, task_c, reorder};
      if (task_c.h0_layer_idx == 0) {
        available_tasks_params_list.push_back(params);
        nof_available_tasks++;
      } else {
        waiting_tasks_params_map[task_c] = params; // Add to map
        nof_waiting_tasks++;
      }
      return eIcicleError::SUCCESS;
    }
    return eIcicleError::INVALID_ARGUMENT;
  }

  template<typename S, typename E>
  bool NttTasksManager<S, E>::get_available_task_to_run(NttTask<S, E>* available_task, int h1_layer) {
    if (!available_tasks_params_list.empty()) {
      // Take the first task from the list
      NttTaskParams<S, E> params = available_tasks_params_list.front();

      // Assign the parameters to the available task
      available_task->set_ntt_cpu(params.ntt_cpu);
      available_task->set_input(params.input);
      available_task->set_coordinates(params.task_c);
      available_task->set_reorder(params.reorder);

      // Remove the task from the list
      available_tasks_params_list.erase(available_tasks_params_list.begin());
      nof_available_tasks--;

      return true;
    }
    return false;
  }


  // Function to set a task as completed and update dependencies
  template<typename S, typename E>
  eIcicleError NttTasksManager<S, E>::set_task_as_completed(NttTask<S, E>& completed_task, int nof_subntts_l2) {
    ntt_cpu::NttTaskCordinates task_c = completed_task.get_coordinates();
    auto& status = tasks_status[task_c.h1_layer_idx][task_c];
    status.done = true;
    // int h1_layer_idx = task_c.h1_layer_idx;
    int nof_h0_layers = counters[task_c.h1_layer_idx].get_nof_h0_layers();
    // Update dependencies in counters
    if(counters[task_c.h1_layer_idx].decrement_counter(task_c)){

      if (task_c.h0_layer_idx < nof_h0_layers-1) {
        int nof_pointing_to_counter = (task_c.h0_layer_idx == nof_h0_layers-1) ? 1
                                      : counters[task_c.h1_layer_idx].get_nof_pointing_to_counter(task_c.h0_layer_idx+1);
        int stride = nof_subntts_l2/nof_pointing_to_counter;
        // int counter_group_idx = task_c.h0_layer_idx==0 ? task_c.h0_block_idx :
        //                       /*task_c.h0_layer_idx==1*/ task_c.h0_subntt_idx;
        for (int i = 0; i < nof_pointing_to_counter; i++) { // TODO - improve efficiency using make_move_iterator
          NttTaskCordinates next_task_c = task_c.h0_layer_idx==0 ?  NttTaskCordinates{task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx+1, task_c.h0_block_idx, i}
                                        /*task_c.h0_layer_idx==1*/: NttTaskCordinates{task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx+1, (task_c.h0_subntt_idx + stride*i), 0};
                                        // /*task_c.h0_layer_idx==1*/: NttTaskCordinates{task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx+1, (task_c.h0_subntt_idx* nof_pointing_to_counter +i), 0};
          if (waiting_tasks_params_map.find(next_task_c) != waiting_tasks_params_map.end()) {
            available_tasks_params_list.push_back(waiting_tasks_params_map[next_task_c]);
            waiting_tasks_params_map.erase(next_task_c);
          }
          else {
            ICICLE_LOG_ERROR << "Task not found in waiting_tasks_params_map: h0_layer_idx: " << next_task_c.h0_layer_idx << ", h0_block_idx: " << next_task_c.h0_block_idx << ", h0_subntt_idx: " << next_task_c.h0_subntt_idx;
          }
        }
          nof_available_tasks = nof_available_tasks + nof_pointing_to_counter;
          nof_waiting_tasks   = nof_waiting_tasks   - nof_pointing_to_counter;

      } else {
        // Reorder the output
        NttTaskCordinates next_task_c = {task_c.h1_layer_idx, task_c.h1_subntt_idx, nof_h0_layers, 0, 0};

        if (waiting_tasks_params_map.find(next_task_c) != waiting_tasks_params_map.end()) {
          available_tasks_params_list.push_back(waiting_tasks_params_map[next_task_c]);
          nof_available_tasks++;
          waiting_tasks_params_map.erase(next_task_c);
          nof_waiting_tasks--;
        }
        else {
          ICICLE_LOG_ERROR << "Task not found in waiting_tasks_params_map";
        }      
      }
    }
    return eIcicleError::SUCCESS;
  }


  //////////////////////////// NttCpu Implementation ////////////////////////////

  template <typename S, typename E>
  int NttCpu<S, E>::bit_reverse(int n, int logn)
  {
    int rev = 0;
    for (int j = 0; j < logn; ++j) {
      if (n & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  }

  template <typename S, typename E>
  // inline uint64_t NttCpu<S, E>::idx_in_mem(
  uint64_t NttCpu<S, E>::idx_in_mem(NttTaskCordinates ntt_task_cordinates, int element)
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
  eIcicleError NttCpu<S, E>::reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy)
  {
    uint64_t subntt_size = is_top_hirarchy ? (1 << (this->ntt_sub_logn.logn)) : 1 << this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int subntt_log_size = is_top_hirarchy ? (this->ntt_sub_logn.logn) : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
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
  void NttCpu<S, E>::dit_ntt( E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles) // R --> N
  {
    uint64_t subntt_size = 1 << this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * (1<<this->ntt_sub_logn.logn);
      for (int len = 2; len <= subntt_size; len <<= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (this->domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, i + j);
            uint64_t v_mem_idx =
              stride * this->idx_in_mem(ntt_task_cordinates, i + j + half_len);
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
  void NttCpu<S, E>::dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles)
  {
    uint64_t subntt_size = 1 << this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * (1<<this->ntt_sub_logn.logn);
      for (int len = subntt_size; len >= 2; len >>= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (this->domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, i + j);
            uint64_t v_mem_idx =
              stride * this->idx_in_mem(ntt_task_cordinates, i + j + half_len);
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
  eIcicleError NttCpu<S, E>::coset_mul(E* elements, const S* twiddles, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset)
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
          current_elements[i] = current_elements[i] * arbitrary_coset[idx];
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
  eIcicleError NttCpu<S, E>::reorder_input(E* input)
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
  void NttCpu<S, E>::refactor_and_reorder(E* elements, const S* twiddles)
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
      E* cur_temp_elements = this->config.columns_batch ? temp_elements.get() + batch : temp_elements.get() + batch * ntt_size;
      for (int sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (int elem = 0; elem < sntt_size; elem++) {
          uint64_t tw_idx = (this->direction == NTTDir::kForward)
                              ? ((this->domain_max_size / ntt_size) * sntt_idx * elem)
                              : this->domain_max_size - ((this->domain_max_size / ntt_size) * sntt_idx * elem);
          cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
            cur_layer_output[stride * (elem * nof_sntts + sntt_idx)] * twiddles[tw_idx];
        }
      }
    }
    std::copy(temp_elements.get(), temp_elements.get() + temp_elements_size, elements);
  }

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::reorder_output(E* output, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy)
  { // TODO shanie future - consider using an algorithm for efficient reordering
    bool is_only_h0 = this->ntt_sub_logn.h1_layers_sub_logn[0] == 0;
    uint64_t size = (is_top_hirarchy || is_only_h0)? 1 << this->ntt_sub_logn.logn 
                                                   : 1 << this->ntt_sub_logn.h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx];
    uint64_t temp_output_size = this->config.columns_batch ? size * this->config.batch_size : size;
    auto temp_output = std::make_unique<E[]>(temp_output_size);
    uint64_t idx = 0;
    uint64_t mem_idx = 0;
    uint64_t new_idx = 0;
    int subntt_idx;
    int element;
    int s0 = is_top_hirarchy? this->ntt_sub_logn.h1_layers_sub_logn[0] : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
    int s1 = is_top_hirarchy? this->ntt_sub_logn.h1_layers_sub_logn[1] : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
    int s2 = is_top_hirarchy? this->ntt_sub_logn.h1_layers_sub_logn[2] : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
    int p0, p1, p2;
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    int rep = this->config.columns_batch ? this->config.batch_size : 1;
    E* h1_subntt_output =
    output + stride * (ntt_task_cordinates.h1_subntt_idx << NttCpu<S, E>::ntt_sub_logn.h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size //TODO - NttCpu or this?
    for (int batch = 0; batch < rep; ++batch) {
      E* current_elements =
        this->config.columns_batch
          ? h1_subntt_output + batch
          : h1_subntt_output; // if columns_batch=false, then output is already shifted by batch*size when calling the function
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
  void NttCpu<S, E>::refactor_output_h0(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles)
  {
    int h0_subntt_size = 1 << NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int h0_nof_subntts = 1 << NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]; //only relevant for layer 1 
    int i, j, i_0;
    int ntt_size = ntt_task_cordinates.h0_layer_idx == 0 ? 1 << (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] + NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1])
                                                : 1 << (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] + NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1] + NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]);
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t original_size = (1 << NttCpu<S, E>::ntt_sub_logn.logn);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* h1_subntt_elements =
      elements + stride * (ntt_task_cordinates.h1_subntt_idx << NttCpu<S, E>::ntt_sub_logn.h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size
      E* elements_of_current_batch = this->config.columns_batch ? h1_subntt_elements + batch : h1_subntt_elements + batch * original_size;
      for (int elem = 0; elem < h0_subntt_size; elem++) {
        uint64_t elem_mem_idx = stride * this->idx_in_mem(ntt_task_cordinates, elem);
        i = (ntt_task_cordinates.h0_layer_idx == 0) ? elem : elem * h0_nof_subntts + ntt_task_cordinates.h0_subntt_idx;
        j = (ntt_task_cordinates.h0_layer_idx == 0) ? ntt_task_cordinates.h0_subntt_idx : ntt_task_cordinates.h0_block_idx;
        uint64_t tw_idx = (this->direction == NTTDir::kForward) ? ((this->domain_max_size / ntt_size) * j * i)
                                                    : this->domain_max_size - ((this->domain_max_size / ntt_size) * j * i);
        // if (ntt_task_cordinates.h0_layer_idx == 1){
        //   ICICLE_LOG_DEBUG << "elem_mem_idx: " << elem_mem_idx << ", i: " << i << ", j: " << j << ", tw_idx: " << tw_idx;
        // }
        elements_of_current_batch[elem_mem_idx] = elements_of_current_batch[elem_mem_idx] * twiddles[tw_idx];
      }
    }
  }

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::h1_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates, NttTasksManager<S, E>& ntt_tasks_manager)
  {
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    int nof_h0_layers = (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2] != 0) ? 3 :
                        (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1] != 0) ? 2 : 1;
    // Assuming that NTT fits in the cache, so we split the NTT to layers and calculate them one after the other.
    // Subntts inside the same laye are calculate in parallel.
    // Sorting is not needed, since the elements needed for each subntt are close to each other in memory.
    // Instead of sorting, we are using the function idx_in_mem to calculate the memory index of each element.
    for (ntt_task_cordinates.h0_layer_idx = 0; ntt_task_cordinates.h0_layer_idx < NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx].size(); ntt_task_cordinates.h0_layer_idx++) {
      if (ntt_task_cordinates.h0_layer_idx == 0) {
        // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx;
        int log_nof_subntts = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
        int log_nof_blocks = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks); ntt_task_cordinates.h0_block_idx++) {
          for ( ntt_task_cordinates.h0_subntt_idx = 0; ntt_task_cordinates.h0_subntt_idx < (1 << log_nof_subntts); ntt_task_cordinates.h0_subntt_idx++) {
            // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
            ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
            // this->h0_cpu_ntt(input, ntt_task_cordinates);
          }
        }
      }
      if (ntt_task_cordinates.h0_layer_idx == 1 && NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]) {
        // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx;
        int log_nof_subntts = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
        int log_nof_blocks = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks); ntt_task_cordinates.h0_block_idx++) {
          for (ntt_task_cordinates.h0_subntt_idx = 0; ntt_task_cordinates.h0_subntt_idx < (1 << log_nof_subntts); ntt_task_cordinates.h0_subntt_idx++) {
            ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
            // this->h0_cpu_ntt(input, ntt_task_cordinates); // input=output (in-place)
          }
        }
      }
      if (ntt_task_cordinates.h0_layer_idx == 2 && NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]) {
        ntt_task_cordinates.h0_subntt_idx = 0; // not relevant for layer 2
        // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx;
        int log_nof_blocks = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] + NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks); ntt_task_cordinates.h0_block_idx++) {
          // ICICLE_LOG_DEBUG << "h1_cpu_ntt: h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx;
          ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
          // this->h0_cpu_ntt(input, ntt_task_cordinates);
        }
      }
    }
    // Sort the output at the end so that elements will be in right order.
    // TODO SHANIE  - After implementing for different ordering, maybe this should be done in a different place
    //              - When implementing real parallelism, consider sorting in parallel and in-place
    int nof_h0_subntts = (nof_h0_layers == 1) ? (1 << NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]) : 
                         (nof_h0_layers == 2) ? (1 << NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]) : 1;
    int nof_h0_blocks  = (nof_h0_layers != 3) ? (1 << NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]) : (1 << (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]+NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]));
    if(nof_h0_layers>1){
      // ICICLE_LOG_DEBUG << "h1_cpu_ntt: PUSH REORDER TASK h0_layer_idx: " << ntt_task_cordinates.h0_layer_idx << ", h0_block_idx: " << ntt_task_cordinates.h0_block_idx << ", h0_subntt_idx: " << ntt_task_cordinates.h0_subntt_idx;
      ntt_task_cordinates = {ntt_task_cordinates.h1_layer_idx, ntt_task_cordinates.h1_subntt_idx, nof_h0_layers, 0, 0};
      ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, true); //reorder=true
      // ICICLE_LOG_DEBUG << "h1_cpu_ntt: PUSH REORDER TASK DONE";
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
  eIcicleError NttCpu<S, E>::h0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates)
  {
    const uint64_t subntt_size = (1 << NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx]);
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    const uint64_t total_memory_size = original_size * this->config.batch_size;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    int offset = this->config.columns_batch ? this->config.batch_size : 1;
    E* current_input =
      input + offset * (ntt_task_cordinates.h1_subntt_idx << NttCpu<S, E>::ntt_sub_logn.h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size

    this->reorder_by_bit_reverse(ntt_task_cordinates, current_input, false); // TODO - check if access the fixed indexes instead of reordering may be more efficient?

    // NTT/INTT
    this->dit_ntt(current_input, ntt_task_cordinates, twiddles); // R --> N

    if (ntt_task_cordinates.h0_layer_idx != 2 && this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx + 1] != 0) {
      this->refactor_output_h0(input, ntt_task_cordinates, twiddles);
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::handle_pushed_tasks(TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, int h1_layer_idx) {
    NttTask<S, E>* task_slot = nullptr;
    int nof_subntts_l2 = 1 << ((this->ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]) + (this->ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1]));
    // ICICLE_LOG_DEBUG << "handle_pushed_tasks: nof_available_tasks: " << ntt_tasks_manager.nof_available_tasks;
    // ICICLE_LOG_DEBUG << "handle_pushed_tasks: nof_waiting_tasks: " << ntt_tasks_manager.nof_waiting_tasks;
    while (ntt_tasks_manager.nof_available_tasks > 0 || ntt_tasks_manager.nof_waiting_tasks > 0) {
      if (ntt_tasks_manager.nof_available_tasks > 0){
        task_slot = tasks_manager->get_idle_or_completed_task();
        if (task_slot->is_completed()) {
          ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l2);
        }
        ntt_tasks_manager.get_available_task_to_run(task_slot, h1_layer_idx);
        task_slot->dispatch();
      } else {  // wait for available tasks
        while (ntt_tasks_manager.nof_available_tasks == 0 && ntt_tasks_manager.nof_waiting_tasks > 0) {
          task_slot = tasks_manager->get_completed_task();
          ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l2);
          if (ntt_tasks_manager.nof_available_tasks > 0) {
            ICICLE_ASSERT(ntt_tasks_manager.get_available_task_to_run(task_slot, h1_layer_idx));
            task_slot->dispatch();
          } else {
            task_slot->set_idle();
          }
        }
      }
    }
    while ((task_slot = tasks_manager->get_completed_task()) != nullptr) { // clean all completed tasks
      task_slot->set_idle();
    }
    return eIcicleError::SUCCESS;
  }

} // namespace ntt_cpu