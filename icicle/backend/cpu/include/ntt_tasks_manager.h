#pragma once
#include "ntt_task.h"


using namespace field_config;
using namespace icicle;
namespace ntt_cpu {


  /**
   * @brief Manages task dependency counters for NTT computation, tracking readiness of tasks to execute.
   *
   * This class tracks and manages counters for tasks within the NTT hierarchy, determining when tasks are ready to
   * execute based on the completion of their dependencies.
   *
   * @param hierarchy_1_layer_idx Index of the hierarchy_1 layer this counter set belongs to.
   * @param nof_hierarchy_0_layers Number of hierarchy_0 layers in the current hierarchy_1 layer.
   * @param dependent_subntt_count Number of counters pointing to each hierarchy_0 layer.
   * @param hierarchy_0_counters A 3D vector of uint32_t - counters for groups of sub-NTTs in hierarchy_0 layers.
   * @param hierarchy_1_counters A vector of shared pointers to counters for each sub-NTT in hierarchy_1 layers, used to
   * signal when an hierarchy_1_subntt is ready for reordering.
   *
   * @method TasksDependenciesCounters(NttSubLogn ntt_sub_logn, uint32_t hierarchy_1_layer_idx) Constructor that initializes
   * the counters based on NTT structure.
   * @method bool decrement_counter(NttTaskCordinates ntt_task_cordinates) Decrements the counter for a given task and
   * returns true if the task is ready to execute.
   * @method uint32_t get_dependent_subntt_count(uint32_t hierarchy_0_layer_idx) Returns the number of counters pointing to the
   * given hierarchy_0 layer.
   * @method uint32_t get_nof_hierarchy_0_layers() Returns the number of hierarchy_0 layers in the current hierarchy_1 layer.
   */
  class TasksDependenciesCounters
  {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCounters(const NttSubLogn& ntt_sub_logn, uint32_t hierarchy_1_layer_idx);

    // Function to decrement the counter for a given task and check if it is ready to execute. if so, return true
    bool decrement_counter(NttTaskCordinates ntt_task_cordinates);
    uint32_t get_dependent_subntt_count(uint32_t hierarchy_0_layer_idx) { return dependent_subntt_count[hierarchy_0_layer_idx]; }
    uint32_t get_nof_hierarchy_0_layers() { return nof_hierarchy_0_layers; }

  private:
    uint32_t hierarchy_1_layer_idx;
    uint32_t nof_hierarchy_0_layers;
    std::vector<uint32_t> dependent_subntt_count; // Number of subntt that are getting available together when a group of
                                             // hierarchy_0_subntts from previous layer are done

    // Each hierarchy_1_subntt has its own set of counters
    std::vector<std::vector<std::vector<uint32_t>>>
      hierarchy_0_counters; // [hierarchy_1_subntt_idx][hierarchy_0_layer_idx][hierarchy_0_counter_idx]

    // One counter for each hierarchy_1_subntt to signal the end of the hierarchy_1_subntt. each hierarchy_0_subntt of
    // last hierarchy_0_layer will decrement this counter when it finishes and when it reaches 0, the hierarchy_1_subntt
    // is ready to reorder
    std::vector<uint32_t> hierarchy_1_counters; // [hierarchy_1_subntt_idx]
  };


  /**
   * @brief Manages tasks for the NTT computation, handling task scheduling and dependency management.
   *
   * The NttTasksManager is responsible for adding tasks, updating task dependencies,
   * and determining the readiness of tasks for execution. It maintains a list of available tasks
   * and a map of tasks that are waiting for dependencies to be resolved.
   *
   */

  template <typename S = scalar_t, typename E = scalar_t>
  class NttTasksManager 
  {
  public:
    NttTasksManager(const NttSubLogn& ntt_sub_logn_ref, uint32_t logn);

    // Add a new task to the ntt_task_manager
    eIcicleError push_task(NttTaskCordinates ntt_task_cordinates);

    // Set a task as completed and update dependencies
    eIcicleError set_task_as_completed(NttTask<S, E>& completed_task, uint32_t nof_subntts_l1);

    bool tasks_to_do() { return !available_tasks_list.empty() || !waiting_tasks_list.empty(); }
    // bool tasks_to_do() { return !available_tasks_list.empty() || nof_pending_tasks!=0; }

    bool available_tasks() { return !available_tasks_list.empty(); }

    NttTaskCordinates get_available_task() { return available_tasks_list.front(); } // //TODO - return pointers to taskscordinates instead of copying

    eIcicleError erase_task_from_available_tasks_list()
    {
      available_tasks_list.pop_front();
      return eIcicleError::SUCCESS;
    }

  private:
    const NttSubLogn& ntt_sub_logn; // Reference to NttSubLogn
    std::vector<TasksDependenciesCounters> counters;      // Dependencies counters by layer
    std::deque<NttTaskCordinates> available_tasks_list; // List of tasks ready to run
    std::deque<NttTaskCordinates> waiting_tasks_list; // List of tasks waiting for dependencies
    // uint32_t nof_pending_tasks = 0;
  };


  //////////////////////////// TasksDependenciesCounters Implementation ////////////////////////////

  /**
   * @brief Initializes dependency counters for NTT task management.
   *
   * This constructor sets up the dependency counters for the different layers within a sub-NTT hierarchy.
   * It configures the counters for hierarchy 0 layers based on the problem size and structure, and initializes
   * counters for each sub-NTT in hierarchy 1. These counters are used to manage task dependencies, ensuring that
   * tasks only execute when their prerequisites are complete.
   *
   * @param ntt_sub_logn The structure containing logarithmic sizes of sub-NTTs.
   * @param hierarchy_1_layer_idx The index of the current hierarchy 1 layer.
   */
  TasksDependenciesCounters::TasksDependenciesCounters(const NttSubLogn& ntt_sub_logn, uint32_t hierarchy_1_layer_idx)
      : hierarchy_0_counters(
          1 << ntt_sub_logn.hierarchy_1_layers_sub_logn
                 [1 - hierarchy_1_layer_idx]), // nof_hierarchy_1_subntts =
                                               // hierarchy_1_layers_sub_logn[1-hierarchy_1_layer_idx].
        hierarchy_1_counters(1 << ntt_sub_logn.hierarchy_1_layers_sub_logn[1 - hierarchy_1_layer_idx])
  {
    nof_hierarchy_0_layers = ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2]
                               ? 3
                               : (ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1] ? 2 : 1);
    dependent_subntt_count.resize(nof_hierarchy_0_layers);
    dependent_subntt_count[0] = 1;
    uint32_t l1_counter_size;
    uint32_t l2_counter_size;
    uint32_t l1_nof_counters;
    uint32_t l2_nof_counters;
    if (nof_hierarchy_0_layers > 1) {
      // Initialize counters for layer 1 - N2 counters initialized with N1.
      dependent_subntt_count[1] = 1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0];
      l1_nof_counters = 1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2];
      l1_counter_size = 1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1];
    }
    if (nof_hierarchy_0_layers > 2) {
      // Initialize counters for layer 2 - N0 counters initialized with N2.
      dependent_subntt_count[2] = 1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1];
      l2_nof_counters = 1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0];
      l2_counter_size = 1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2];
    }

    for (uint32_t hierarchy_1_subntt_idx = 0;
         hierarchy_1_subntt_idx < (1 << ntt_sub_logn.hierarchy_1_layers_sub_logn[1 - hierarchy_1_layer_idx]);
         ++hierarchy_1_subntt_idx) {
      hierarchy_0_counters[hierarchy_1_subntt_idx].resize(3); // 3 possible layers (0, 1, 2)
      // Initialize counters for layer 0 - 1 counter1 initialized with 0.
      hierarchy_0_counters[hierarchy_1_subntt_idx][0].resize(1);
      hierarchy_0_counters[hierarchy_1_subntt_idx][0][0] =
        0; //[hierarchy_1_subntt_idx][hierarchy_0_layer_idx][hierarchy_0_counter_idx]

      if (nof_hierarchy_0_layers > 1) {
        // Initialize counters for layer 1 - N2 counters initialized with N1.
        hierarchy_0_counters[hierarchy_1_subntt_idx][1].resize(l1_nof_counters);
        for (uint32_t counter_idx = 0; counter_idx < l1_nof_counters; ++counter_idx) {
          hierarchy_0_counters[hierarchy_1_subntt_idx][1][counter_idx] = l1_counter_size;
        }
      }
      if (nof_hierarchy_0_layers > 2) {
        // Initialize counters for layer 2 - N0 counters initialized with N2.
        hierarchy_0_counters[hierarchy_1_subntt_idx][2].resize(l2_nof_counters);
        for (uint32_t counter_idx = 0; counter_idx < l2_nof_counters; ++counter_idx) {
          hierarchy_0_counters[hierarchy_1_subntt_idx][2][counter_idx] = l2_counter_size;
        }
      }
      // Initialize hierarchy_1_counters with N0 * N1
      uint32_t hierarchy_1_counter_size =
        nof_hierarchy_0_layers == 3 ? (1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0]) *
                                        (1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1])
        : nof_hierarchy_0_layers == 2 ? (1 << ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0])
                                      : 0;
      hierarchy_1_counters[hierarchy_1_subntt_idx] = hierarchy_1_counter_size;
    }
  }

  /**
   * @brief Decrements the dependency counter for a given task and checks if the next task is ready to execute.
   *
   * This function decrements the counter associated with a task in hierarchy 0 or the global counter
   * in hierarchy 1. If the counter reaches zero, it indicates that the dependent task is now ready
   * to be executed.
   *
   * @param task_c The coordinates of the task whose counter is to be decremented.
   * @return True if the dependent task is ready to execute, false otherwise.
   */
  bool TasksDependenciesCounters::decrement_counter(NttTaskCordinates task_c)
  {
    if (nof_hierarchy_0_layers == 1) { return false; }
    if (task_c.hierarchy_0_layer_idx < nof_hierarchy_0_layers - 1) {
      // Extract the coordinates from the task
      uint32_t counter_group_idx =
        task_c.hierarchy_0_layer_idx == 0 ? task_c.hierarchy_0_block_idx :
                                          /*task_c.hierarchy_0_layer_idx==1*/ task_c.hierarchy_0_subntt_idx;

      uint32_t& counter_ref =
        hierarchy_0_counters[task_c.hierarchy_1_subntt_idx][task_c.hierarchy_0_layer_idx + 1][counter_group_idx];
      counter_ref--;

      if (counter_ref == 0) { return true; }
    } else {
      // Decrement the counter for the given hierarchy_1_subntt_idx
      uint32_t& hierarchy_1_counter_ref = hierarchy_1_counters[task_c.hierarchy_1_subntt_idx];
      hierarchy_1_counter_ref--;

      if (hierarchy_1_counter_ref == 0) { return true; }
    }
    return false;
  }


  //////////////////////////// NttTasksManager Implementation ////////////////////////////

  /**
   * @brief Constructs the task manager for a given problem size.
   * @param logn The log2(size) of the NTT problem.
   */
  template <typename S, typename E>
  NttTasksManager<S, E>::NttTasksManager(const NttSubLogn& ntt_sub_logn_ref, uint32_t logn)
      : ntt_sub_logn(ntt_sub_logn_ref),
        counters(logn > HIERARCHY_1 ? 2 : 1, TasksDependenciesCounters(ntt_sub_logn_ref, 0))
  {
    if (logn > HIERARCHY_1) { counters[1] = TasksDependenciesCounters(ntt_sub_logn_ref, 1); }
  }

  /**
   * @brief Adds a new task to the task manager.
   * @param ntt_task_cordinates Task coordinates specifying the task's position in the hierarchy.
   * @return Status indicating success or failure.
   */
  template <typename S, typename E>
  eIcicleError NttTasksManager<S, E>::push_task(NttTaskCordinates ntt_task_cordinates)
  {
    if (ntt_task_cordinates.hierarchy_0_layer_idx == 0) {
      available_tasks_list.push_back(ntt_task_cordinates);
    } else {
      waiting_tasks_list.push_back(ntt_task_cordinates);
      // nof_pending_tasks++;
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Marks a task as completed and updates dependencies.
   * @param completed_task The completed task.
   * @param nof_subntts_l1 Number of sub-NTTs in the second layer of hierarchy 1.
   * @return Status indicating success or failure.
   */
  template <typename S, typename E>
  eIcicleError NttTasksManager<S, E>::set_task_as_completed(NttTask<S, E>& completed_task, uint32_t nof_subntts_l1)
  {
    NttTaskCordinates task_c = completed_task.get_coordinates();
    uint32_t nof_hierarchy_0_layers = counters[task_c.hierarchy_1_layer_idx].get_nof_hierarchy_0_layers();
    // Update dependencies in counters
    if (counters[task_c.hierarchy_1_layer_idx].decrement_counter(task_c)) {
      if (task_c.hierarchy_0_layer_idx < nof_hierarchy_0_layers - 1) {
        uint32_t dependent_subntt_count = (task_c.hierarchy_0_layer_idx == nof_hierarchy_0_layers - 1) ? 1 : counters[task_c.hierarchy_1_layer_idx].get_dependent_subntt_count(task_c.hierarchy_0_layer_idx + 1);
        uint32_t stride = nof_subntts_l1 / dependent_subntt_count;
        for (uint32_t i = 0; i < dependent_subntt_count; i++) {
          NttTaskCordinates next_task_c = task_c.hierarchy_0_layer_idx == 0 ? NttTaskCordinates{task_c.hierarchy_1_layer_idx, task_c.hierarchy_1_subntt_idx, task_c.hierarchy_0_layer_idx + 1, task_c.hierarchy_0_block_idx, i} 
                                        /*task_c.hierarchy_0_layer_idx==1*/ : NttTaskCordinates{ task_c.hierarchy_1_layer_idx, task_c.hierarchy_1_subntt_idx, task_c.hierarchy_0_layer_idx + 1, (task_c.hierarchy_0_subntt_idx + stride * i), 0};
          // available_tasks_list.push_back(next_task_c);
          // nof_pending_tasks--;
          auto it = std::find(waiting_tasks_list.begin(), waiting_tasks_list.end(), next_task_c); //TODO - try without waiting_tasks_list, just push the task. if it works, remove waiting_tasks_list logic.
          if (it != waiting_tasks_list.end()) {
              available_tasks_list.push_back(*it);
              waiting_tasks_list.erase(it);
          }
        }
      } else {
        // Reorder the output
        NttTaskCordinates next_task_c = {task_c.hierarchy_1_layer_idx, task_c.hierarchy_1_subntt_idx, nof_hierarchy_0_layers, 0, 0, true};
        // available_tasks_list.push_back(next_task_c);
        // nof_pending_tasks--;
        auto it = std::find(waiting_tasks_list.begin(), waiting_tasks_list.end(), next_task_c);
        if (it != waiting_tasks_list.end()) {
            available_tasks_list.push_back(*it);
            waiting_tasks_list.erase(it);
        }
      }
    }
    return eIcicleError::SUCCESS;
  }
} // namespace ntt_cpu

