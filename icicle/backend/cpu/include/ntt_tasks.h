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
#include <deque>
#include <functional>
#include <unordered_map>

#define HIERARCHY_1 15

using namespace field_config;
using namespace icicle;
namespace ntt_cpu {

  /**
   * @brief Defines the log sizes of sub-NTTs for different problem sizes.
   *
   * `layers_sub_logn` specifies the log sizes for up to three layers (hierarchy1 or hierarchy0) in the NTT computation.
   * - The outer index represents the log size (`logn`) of the original NTT problem.
   * - Each inner array contains three integers corresponding to the log sizes for each hierarchical layer.
   *
   * Example: `layers_sub_logn[14] = {14, 13, 0}` means for `logn = 14`, the sub-NTT log sizes are 14 for the first
   * layer, 13 for the second, and 0 for the third.
   */
  constexpr uint32_t layers_sub_logn[31][3] = {
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

  /**
   * @brief Represents the coordinates of a task in the NTT hierarchy.
   * This struct holds indices that identify the position of a task within the NTT computation hierarchy.
   *
   * @param hierarchy_1_layer_idx Index of the hierarchy_1 layer.
   * @param hierarchy_1_subntt_idx Index of the sub-NTT within the hierarchy_1 layer.
   * @param hierarchy_0_layer_idx Index of the hierarchy_0 layer.
   * @param hierarchy_0_block_idx Index of the block within the hierarchy_0 layer.
   * @param hierarchy_0_subntt_idx Index of the sub-NTT within the hierarchy_0 block.
   *
   * @method bool operator==(const NttTaskCordinates& other) const Compares two task coordinates for equality.
   */
  struct NttTaskCordinates {
    int hierarchy_1_layer_idx = 0;
    int hierarchy_1_subntt_idx = 0;
    int hierarchy_0_layer_idx = 0;
    int hierarchy_0_block_idx = 0;
    int hierarchy_0_subntt_idx = 0;

    bool operator==(const NttTaskCordinates& other) const
    {
      return hierarchy_1_layer_idx == other.hierarchy_1_layer_idx &&
             hierarchy_1_subntt_idx == other.hierarchy_1_subntt_idx &&
             hierarchy_0_layer_idx == other.hierarchy_0_layer_idx &&
             hierarchy_0_block_idx == other.hierarchy_0_block_idx &&
             hierarchy_0_subntt_idx == other.hierarchy_0_subntt_idx;
    }
  };

  /**
   * @brief Represents the log sizes of sub-NTTs in the NTT computation hierarchy.
   *
   * This struct stores the log sizes of the sub-NTTs for both hierarchy_0 and hierarchy_1  layers,
   * based on the overall log size (`logn`) of the NTT problem.
   *
   * @param logn The log size of the entire NTT problem.
   * @param size The size of the NTT problem, calculated as `1 << logn`.
   * @param hierarchy_0_layers_sub_logn Log sizes of sub-NTTs for hierarchy_0 layers.
   * @param hierarchy_1_layers_sub_logn Log sizes of sub-NTTs for hierarchy_1 layers.
   *
   * @method NttSubLogn(int logn) Initializes the struct based on the given `logn`.
   */
  struct NttSubLogn {
    int logn;                                                  // Original log_size of the problem
    uint64_t size;                                             // Original log_size of the problem
    std::vector<std::vector<int>> hierarchy_0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<int> hierarchy_1_layers_sub_logn;              // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubLogn(int logn) : logn(logn)
    {
      size = 1 << logn;
      if (logn > HIERARCHY_1) {
        // Initialize hierarchy_1_layers_sub_logn
        hierarchy_1_layers_sub_logn =
          std::vector<int>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn]));
        // Initialize hierarchy_0_layers_sub_logn
        hierarchy_0_layers_sub_logn = {
          std::vector<int>(
            std::begin(layers_sub_logn[hierarchy_1_layers_sub_logn[0]]),
            std::end(layers_sub_logn[hierarchy_1_layers_sub_logn[0]])),
          std::vector<int>(
            std::begin(layers_sub_logn[hierarchy_1_layers_sub_logn[1]]),
            std::end(layers_sub_logn[hierarchy_1_layers_sub_logn[1]]))};
      } else {
        hierarchy_1_layers_sub_logn = {0, 0, 0};
        hierarchy_0_layers_sub_logn = {
          std::vector<int>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn])), {0, 0, 0}};
      }
    }
  };

  template <typename S, typename E>
  class NttTask;

  template <typename S, typename E>
  class NttTasksManager;

  /**
   * @brief Manages the Number Theoretic Transform (NTT) computation, including various transformation and reordering
   * tasks.
   *
   * This class handles the full process of NTT computation, including initialization, transformations,
   * and handling task dependencies within a hierarchical structure.
   *
   * @param ntt_sub_logn Stores the log sizes for sub-NTTs in different layers.
   * @param direction Indicates the direction of the NTT (forward or inverse).
   * @param config Configuration settings for the NTT.
   * @param domain_max_size Maximum size of the NTT domain.
   * @param twiddles Pointer to the twiddle factors used in the NTT.
   */
  template <typename S = scalar_t, typename E = scalar_t>
  class NttCpu
  {
  public:
    NttSubLogn ntt_sub_logn;
    NTTDir direction;
    const NTTConfig<S>& config;
    int domain_max_size;
    const S* twiddles;

    NttCpu(int logn, NTTDir direction, const NTTConfig<S>& config, int domain_max_size, const S* twiddles)
        : ntt_sub_logn(logn), direction(direction), config(config), domain_max_size(domain_max_size), twiddles(twiddles)
    {
    }

    void reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy);
    eIcicleError copy_and_reorder_if_needed(const E* input, E* output);
    void hierarchy_0_dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates);
    void hierarchy_0_dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates);
    void coset_mul(E* elements, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset);
    int find_or_generate_coset(std::unique_ptr<S[]>& arbitrary_coset);
    void hierarchy_1_reorder(E* elements);
    eIcicleError
    reorder_and_refactor_if_needed(E* elements, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy);
    eIcicleError
    hierarchy1_push_tasks(E* input, NttTaskCordinates ntt_task_cordinates, NttTasksManager<S, E>& ntt_tasks_manager);
    eIcicleError hierarchy_0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates);
    eIcicleError handle_pushed_tasks(
      TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, int hierarchy_1_layer_idx);

  private:
    uint64_t bit_reverse(uint64_t n, int logn);
    uint64_t idx_in_mem(NttTaskCordinates ntt_task_cordinates, int element);
    void refactor_output_hierarchy_0(E* elements, NttTaskCordinates ntt_task_cordinates);
  }; // class NttCpu

  /**
   * @brief Manages task dependency counters for NTT computation, tracking readiness of tasks to execute.
   *
   * This class tracks and manages counters for tasks within the NTT hierarchy, determining when tasks are ready to
   * execute based on the completion of their dependencies.
   *
   * @param hierarchy_1_layer_idx Index of the hierarchy_1 layer this counter set belongs to.
   * @param nof_hierarchy_0_layers Number of hierarchy_0 layers in the current hierarchy_1 layer.
   * @param dependent_subntt_count Number of counters pointing to each hierarchy_0 layer.
   * @param hierarchy_0_counters A 3D vector of int - counters for groups of sub-NTTs in hierarchy_0 layers.
   * @param hierarchy_1_counters A vector of shared pointers to counters for each sub-NTT in hierarchy_1 layers, used to
   * signal when an hierarchy_1_subntt is ready for reordering.
   *
   * @method TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int hierarchy_1_layer_idx) Constructor that initializes
   * the counters based on NTT structure.
   * @method bool decrement_counter(NttTaskCordinates ntt_task_cordinates) Decrements the counter for a given task and
   * returns true if the task is ready to execute.
   * @method int get_dependent_subntt_count(int hierarchy_0_layer_idx) Returns the number of counters pointing to the
   * given hierarchy_0 layer.
   * @method int get_nof_hierarchy_0_layers() Returns the number of hierarchy_0 layers in the current hierarchy_1 layer.
   */
  class TasksDependenciesCounters
  {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int hierarchy_1_layer_idx);

    // Function to decrement the counter for a given task and check if it is ready to execute. if so, return true
    bool decrement_counter(NttTaskCordinates ntt_task_cordinates);
    int get_dependent_subntt_count(int hierarchy_0_layer_idx) { return dependent_subntt_count[hierarchy_0_layer_idx]; }
    int get_nof_hierarchy_0_layers() { return nof_hierarchy_0_layers; }

  private:
    int hierarchy_1_layer_idx;
    int nof_hierarchy_0_layers;
    std::vector<int> dependent_subntt_count; // Number of subntt that are getting available together when a group of
                                             // hierarchy_0_subntts from previous layer are done

    // Each hierarchy_1_subntt has its own set of counters
    std::vector<std::vector<std::vector<int>>>
      hierarchy_0_counters; // [hierarchy_1_subntt_idx][hierarchy_0_layer_idx][hierarchy_0_counter_idx]

    // One counter for each hierarchy_1_subntt to signal the end of the hierarchy_1_subntt. each hierarchy_0_subntt of
    // last hierarchy_0_layer will decrement this counter when it finishes and when it reaches 0, the hierarchy_1_subntt
    // is ready to reorder
    std::vector<int> hierarchy_1_counters; // [hierarchy_1_subntt_idx]
  };

  template <typename S = scalar_t, typename E = scalar_t>
  struct NttTaskParams {
    NttCpu<S, E>* ntt_cpu;
    E* input;
    NttTaskCordinates task_c;
    bool reorder;
  };

  /**
   * @brief Represents a task in the NTT computation, handling either NTT calculation or reordering.
   *
   * This class manages tasks within the NTT computation, performing either the NTT computation
   * for a given sub-NTT or reordering the output if required.
   *
   * @tparam S Scalar type.
   * @tparam E Element type.
   *
   * @param ntt_cpu Pointer to the NttCpu instance managing the task.
   * @param input Pointer to the input data for the task.
   * @param ntt_task_cordinates Coordinates specifying the task's position within the NTT hierarchy.
   * @param reorder Flag indicating whether the task involves reordering.

   * @method void execute() Executes the task, either performing the NTT computation or reordering the output.
   * @method NttTaskCordinates get_coordinates() const Returns the task's coordinates.
   * @method bool is_reorder() const Checks if the task is a reorder task.
   * @method void set_params(NttTaskParams<S, E> params) Sets the task parameters.
   */
  template <typename S = scalar_t, typename E = scalar_t>
  class NttTask : public TaskBase
  {
  public:
    NttTask() : ntt_cpu(nullptr), input(nullptr), reorder(false) {}

    void execute()
    {
      if (reorder) {
        // if all hierarchy_0_subntts are done, and at least 2 layers in hierarchy 0 - reorder the subntt's output
        if (ntt_cpu->config.columns_batch) {
          ntt_cpu->reorder_and_refactor_if_needed(input, ntt_task_cordinates, false);
        } else {
          for (int b = 0; b < ntt_cpu->config.batch_size; b++) {
            ntt_cpu->reorder_and_refactor_if_needed(
              input + b * (1 << (ntt_cpu->ntt_sub_logn.logn)), ntt_task_cordinates, false);
          }
        }
      } else {
        ntt_cpu->hierarchy_0_cpu_ntt(input, ntt_task_cordinates);
      }
    }

    NttTaskCordinates get_coordinates() const { return ntt_task_cordinates; }

    bool is_reorder() const { return reorder; }
    void set_params(NttTaskParams<S, E> params)
    {
      ntt_cpu = params.ntt_cpu;
      input = params.input;
      ntt_task_cordinates = params.task_c;
      reorder = params.reorder;
    }

  private:
    NttCpu<S, E>* ntt_cpu; // Reference to NttCpu instance
    E* input;
    NttTaskCordinates ntt_task_cordinates;
    bool reorder;
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
    NttTasksManager(int logn);

    // Add a new task to the ntt_task_manager
    eIcicleError push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder);

    // Set a task as completed and update dependencies
    eIcicleError set_task_as_completed(NttTask<S, E>& completed_task, int nof_subntts_l1);

    bool tasks_to_do() { return !available_tasks_list.empty() || !waiting_tasks_map.empty(); }

    bool available_tasks() { return !available_tasks_list.empty(); }

    NttTaskParams<S, E> get_available_task() { return available_tasks_list.front(); }

    eIcicleError erase_task_from_available_tasks_list()
    {
      available_tasks_list.pop_front();
      return eIcicleError::SUCCESS;
    }

  private:
    std::vector<TasksDependenciesCounters> counters;      // Dependencies counters by layer
    std::deque<NttTaskParams<S, E>> available_tasks_list; // List of tasks ready to run
    std::unordered_map<NttTaskCordinates, NttTaskParams<S, E>>
      waiting_tasks_map; // Map of tasks waiting for dependencies
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
  TasksDependenciesCounters::TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int hierarchy_1_layer_idx)
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
    int l1_counter_size;
    int l2_counter_size;
    int l1_nof_counters;
    int l2_nof_counters;
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

    for (int hierarchy_1_subntt_idx = 0;
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
        for (int counter_idx = 0; counter_idx < l1_nof_counters; ++counter_idx) {
          hierarchy_0_counters[hierarchy_1_subntt_idx][1][counter_idx] = l1_counter_size;
        }
      }
      if (nof_hierarchy_0_layers > 2) {
        // Initialize counters for layer 2 - N0 counters initialized with N2.
        hierarchy_0_counters[hierarchy_1_subntt_idx][2].resize(l2_nof_counters);
        for (int counter_idx = 0; counter_idx < l2_nof_counters; ++counter_idx) {
          hierarchy_0_counters[hierarchy_1_subntt_idx][2][counter_idx] = l2_counter_size;
        }
      }
      // Initialize hierarchy_1_counters with N0 * N1
      int hierarchy_1_counter_size =
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
      int counter_group_idx =
        task_c.hierarchy_0_layer_idx == 0 ? task_c.hierarchy_0_block_idx :
                                          /*task_c.hierarchy_0_layer_idx==1*/ task_c.hierarchy_0_subntt_idx;

      int& counter_ref =
        hierarchy_0_counters[task_c.hierarchy_1_subntt_idx][task_c.hierarchy_0_layer_idx + 1][counter_group_idx];
      counter_ref--;

      if (counter_ref == 0) { return true; }
    } else {
      // Decrement the counter for the given hierarchy_1_subntt_idx
      int& hierarchy_1_counter_ref = hierarchy_1_counters[task_c.hierarchy_1_subntt_idx];
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
  NttTasksManager<S, E>::NttTasksManager(int logn)
      : counters(logn > HIERARCHY_1 ? 2 : 1, TasksDependenciesCounters(NttSubLogn(logn), 0))
  {
    if (logn > HIERARCHY_1) { counters[1] = TasksDependenciesCounters(NttSubLogn(logn), 1); }
  }

  /**
   * @brief Adds a new task to the task manager.
   * @param ntt_cpu Pointer to the NTT CPU instance.
   * @param input Pointer to the input data.
   * @param task_c Task coordinates specifying the task's position in the hierarchy.
   * @param reorder Flag indicating if the task requires reordering.
   * @return Status indicating success or failure.
   */
  template <typename S, typename E>
  eIcicleError NttTasksManager<S, E>::push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder)
  {
    // Create a new NttTaskParams and add it to the available_tasks_list
    NttTaskParams<S, E> params = {ntt_cpu, input, task_c, reorder};
    if (task_c.hierarchy_0_layer_idx == 0) {
      available_tasks_list.push_back(params);
    } else {
      waiting_tasks_map[task_c] = params; // Add to map
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
  eIcicleError NttTasksManager<S, E>::set_task_as_completed(NttTask<S, E>& completed_task, int nof_subntts_l1)
  {
    ntt_cpu::NttTaskCordinates task_c = completed_task.get_coordinates();
    int nof_hierarchy_0_layers = counters[task_c.hierarchy_1_layer_idx].get_nof_hierarchy_0_layers();
    // Update dependencies in counters
    if (counters[task_c.hierarchy_1_layer_idx].decrement_counter(task_c)) {
      if (task_c.hierarchy_0_layer_idx < nof_hierarchy_0_layers - 1) {
        int dependent_subntt_count =
          (task_c.hierarchy_0_layer_idx == nof_hierarchy_0_layers - 1)
            ? 1
            : counters[task_c.hierarchy_1_layer_idx].get_dependent_subntt_count(task_c.hierarchy_0_layer_idx + 1);
        int stride = nof_subntts_l1 / dependent_subntt_count;
        for (int i = 0; i < dependent_subntt_count; i++) {
          NttTaskCordinates next_task_c =
            task_c.hierarchy_0_layer_idx == 0
              ? NttTaskCordinates{task_c.hierarchy_1_layer_idx, task_c.hierarchy_1_subntt_idx, task_c.hierarchy_0_layer_idx + 1, task_c.hierarchy_0_block_idx, i}
              /*task_c.hierarchy_0_layer_idx==1*/
              : NttTaskCordinates{
                  task_c.hierarchy_1_layer_idx, task_c.hierarchy_1_subntt_idx, task_c.hierarchy_0_layer_idx + 1,
                  (task_c.hierarchy_0_subntt_idx + stride * i), 0};
          auto it = waiting_tasks_map.find(next_task_c);
          if (it != waiting_tasks_map.end()) {
            available_tasks_list.push_back(it->second);
            waiting_tasks_map.erase(it);
          }
        }
      } else {
        // Reorder the output
        NttTaskCordinates next_task_c = {
          task_c.hierarchy_1_layer_idx, task_c.hierarchy_1_subntt_idx, nof_hierarchy_0_layers, 0, 0};
        auto it = waiting_tasks_map.find(next_task_c);
        if (it != waiting_tasks_map.end()) {
          available_tasks_list.push_back(it->second);
          waiting_tasks_map.erase(it);
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  //////////////////////////// NttCpu Implementation ////////////////////////////

  /**
   * @brief Computes the bit-reversed value of the given integer, based on the specified number of bits.
   *
   * @param n The integer to be bit-reversed.
   * @param logn The number of bits to consider for the bit-reversal.
   * @return uint64_t The bit-reversed value of `n`.
   */
  template <typename S, typename E>
  uint64_t NttCpu<S, E>::bit_reverse(uint64_t n, int logn)
  {
    int rev = 0;
    for (int j = 0; j < logn; ++j) {
      if (n & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  }

  /**
   * @brief Computes the memory index for a given element based on task coordinates.
   *
   * This function calculates the memory index of an element within the NTT structure
   * based on the provided task coordinates and the current hierarchy layer. The index
   * calculation takes into account that reordering between layers of hierarchy 0 has been
   * skipped, and therefore, the function accesses corrected memory addresses accordingly.
   *
   * The function supports different layer configurations (`hierarchy_0_layer_idx`) within the sub-NTT,
   * and returns the appropriate memory index based on the element's position within the hierarchy.
   *
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT hierarchy.
   * @param element_idx The specific element index within the sub-NTT.
   * @return uint64_t The computed memory index for the given element.
   */

  template <typename S, typename E>
  uint64_t NttCpu<S, E>::idx_in_mem(NttTaskCordinates ntt_task_cordinates, int element_idx)
  {
    int s0 = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0];
    int s1 = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
    int s2 = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
    switch (ntt_task_cordinates.hierarchy_0_layer_idx) {
    case 0:
      return ntt_task_cordinates.hierarchy_0_block_idx +
             ((ntt_task_cordinates.hierarchy_0_subntt_idx + (element_idx << s1)) << s2);
    case 1:
      return ntt_task_cordinates.hierarchy_0_block_idx +
             ((element_idx + (ntt_task_cordinates.hierarchy_0_subntt_idx << s1)) << s2);
    case 2:
      return ((ntt_task_cordinates.hierarchy_0_block_idx << (s1 + s2)) & ((1 << (s0 + s1 + s2)) - 1)) +
             (((ntt_task_cordinates.hierarchy_0_block_idx << (s1 + s2)) >> (s0 + s1 + s2)) << s2) + element_idx;
    default:
      ICICLE_ASSERT(false) << "Unsupported layer";
    }
    return static_cast<uint64_t>(-1);
  }

  /**
   * @brief Reorders elements by bit-reversing their indices within a sub-NTT.
   *
   * This function reorders the elements of a sub-NTT based on the bit-reversed indices.
   * The reordering is performed either on the entire NTT or within a specific sub-NTT,
   * depending on whether the operation is at the top hierarchy level. The function accesses
   * corrected memory addresses, because reordering between layers of hierarchy 0 was skipped.
   *
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT computation.
   * @param elements The array of elements to be reordered.
   * @param is_top_hirarchy Boolean indicating whether the operation is at the top hierarchy level.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy)
  {
    uint64_t subntt_size =
      is_top_hirarchy ? (this->ntt_sub_logn.size)
                      : 1 << this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                                           [ntt_task_cordinates.hierarchy_0_layer_idx];
    int subntt_log_size = is_top_hirarchy
                            ? (this->ntt_sub_logn.logn)
                            : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                                            [ntt_task_cordinates.hierarchy_0_layer_idx];
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * this->ntt_sub_logn.size;
      uint64_t rev;
      uint64_t i_mem_idx;
      uint64_t rev_mem_idx;
      for (uint64_t i = 0; i < subntt_size; ++i) {
        rev = bit_reverse(i, subntt_log_size);
        if (!is_top_hirarchy) {
          i_mem_idx = idx_in_mem(ntt_task_cordinates, i);
          rev_mem_idx = idx_in_mem(ntt_task_cordinates, rev);
        } else {
          i_mem_idx = i;
          rev_mem_idx = rev;
        }
        if (i < rev) {
          if (i_mem_idx < this->ntt_sub_logn.size && rev_mem_idx < this->ntt_sub_logn.size) { // Ensure indices are
                                                                                              // within bounds
            std::swap(current_elements[stride * i_mem_idx], current_elements[stride * rev_mem_idx]);
          } else {
            // Handle out-of-bounds error
            ICICLE_LOG_ERROR << "i=" << i << ", rev=" << rev << ", original_size=" << this->ntt_sub_logn.size;
            ICICLE_LOG_ERROR << "Index out of bounds: i_mem_idx=" << i_mem_idx << ", rev_mem_idx=" << rev_mem_idx;
          }
        }
      }
    }
  }

  /**
   * @brief Copies the input data to the output buffer, applying reordering if necessary.
   *
   * The reordering can involve bit-reversal and/or a custom reorder logic, depending
   * on the size of the input data and the specified ordering configuration. If no reordering
   * is needed, the data is simply copied as-is.
   *
   * @param input The input array from which data will be copied.
   * @param output The output array where the copied (and potentially reordered) data will be stored.
   *
   * @note - If `logn > HIERARCHY_1`, there is an additional level of hierarchy due to sub-NTTs, requiring extra
   * reordering
   *       - If `logn <= HIERARCHY_1`, only bit-reversal reordering is applied if configured.
   *       - If no reordering is needed, the input data is directly copied to the output.
   */

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::copy_and_reorder_if_needed(const E* input, E* output)
  {
    const uint64_t total_memory_size = this->ntt_sub_logn.size * config.batch_size;
    const int stride = config.columns_batch ? config.batch_size : 1;
    const int logn = static_cast<int>(std::log2(this->ntt_sub_logn.size));
    const bool bit_rev = config.ordering == Ordering::kRN || config.ordering == Ordering::kRR;

    // Check if input and output point to the same memory location
    E* temp_output = output;
    std::unique_ptr<E[]> temp_storage;
    if (input == output) {
      // Allocate temporary storage to handle in-place reordering
      temp_storage = std::make_unique<E[]>(total_memory_size);
      temp_output = temp_storage.get();
    }

    if (logn > HIERARCHY_1) {
      // Apply input's reorder logic depending on the configuration
      int cur_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
      int next_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];

      for (int batch = 0; batch < config.batch_size; ++batch) {
        const E* input_batch = config.columns_batch ? (input + batch) : (input + batch * this->ntt_sub_logn.size);
        E* output_batch =
          config.columns_batch ? (temp_output + batch) : (temp_output + batch * this->ntt_sub_logn.size);

        for (uint64_t i = 0; i < this->ntt_sub_logn.size; ++i) {
          int subntt_idx = i >> cur_ntt_log_size;
          int element = i & ((1 << cur_ntt_log_size) - 1);
          uint64_t new_idx = bit_rev ? bit_reverse(subntt_idx + (element << next_ntt_log_size), logn)
                                     : subntt_idx + (element << next_ntt_log_size);
          output_batch[stride * i] = input_batch[stride * new_idx];
        }
      }

    } else if (bit_rev) {
      // Only bit-reverse reordering needed
      for (int batch = 0; batch < config.batch_size; ++batch) {
        const E* input_batch = config.columns_batch ? (input + batch) : (input + batch * this->ntt_sub_logn.size);
        E* output_batch =
          config.columns_batch ? (temp_output + batch) : (temp_output + batch * this->ntt_sub_logn.size);

        for (uint64_t i = 0; i < this->ntt_sub_logn.size; ++i) {
          uint64_t rev = bit_reverse(i, logn);
          output_batch[stride * i] = input_batch[stride * rev];
        }
      }
    } else {
      // Just copy, no reordering needed
      std::copy(input, input + total_memory_size, output);
    }

    if (input == output && (logn > HIERARCHY_1 || bit_rev)) {
      // Copy the reordered elements from the temporary storage back to the output
      std::copy(temp_output, temp_output + total_memory_size, output);
    }

    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Performs the Decimation-In-Time (DIT) NTT transform on a sub-NTT.
   *
   * This function applies the Decimation-In-Time (DIT) Number Theoretic Transform (NTT) to
   * the specified sub-NTT, transforming the data from the bit-reversed order (R) to natural order (N).
   * The transformation is performed iteratively by dividing the sub-NTT into smaller segments, applying
   * butterfly operations, and utilizing twiddle factors.
   *
   * @param elements The array of elements on which to perform the DIT NTT.
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT hierarchy.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::hierarchy_0_dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates) // R --> N
  {
    const int subntt_size_log =
      this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                    [ntt_task_cordinates.hierarchy_0_layer_idx];
    const uint64_t subntt_size = 1 << subntt_size_log;
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);
      for (int len = 2; len <= subntt_size; len <<= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (this->domain_max_size >> subntt_size_log);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = stride * idx_in_mem(ntt_task_cordinates, i + j);
            uint64_t v_mem_idx = stride * idx_in_mem(ntt_task_cordinates, i + j + half_len);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx] * this->twiddles[tw_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = u - v;
          }
        }
      }
    }
  }

  /**
   * @brief Performs the Decimation-In-Frequency (DIF) NTT transform on a sub-NTT.
   *
   * This function applies the Decimation-In-Frequency (DIF) Number Theoretic Transform (NTT)
   * to the specified sub-NTT. The transformation is performed iteratively, starting from the full
   * sub-NTT size and reducing by half at each step, applying butterfly operations and utilizing twiddle factors.
   * transforming the data from the natural order (N) to bit-reversed order (R).
   *
   * @param elements The array of elements on which to perform the DIF NTT.
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT hierarchy.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::hierarchy_0_dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates) // N --> R
  {
    uint64_t subntt_size =
      1 << this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                         [ntt_task_cordinates.hierarchy_0_layer_idx];
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);
      for (int len = subntt_size; len >= 2; len >>= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (this->domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = stride * idx_in_mem(ntt_task_cordinates, i + j);
            uint64_t v_mem_idx = stride * idx_in_mem(ntt_task_cordinates, i + j + half_len);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = (u - v) * this->twiddles[tw_idx];
          }
        }
      }
    }
  }

  /**
   * @brief Applies coset multiplication to the given elements based on the provided twiddles or arbitrary coset.
   *
   * This function performs a coset multiplication on the `elements` array, which may have been reordered
   * due to input reordering operations. The multiplication is based on either a precomputed
   * coset stride using `twiddles` or an `arbitrary_coset` generated dynamically. The function handles
   * different reordering schemes and ensures the correct indices are used for multiplication.
   *
   * @param elements The array of elements to which the coset multiplication will be applied.
   * @param coset_stride The stride used to select the appropriate twiddle factor. This is computed based on the coset
   * generator.
   * @param arbitrary_coset A unique pointer to an array of arbitrary coset values generated if the coset generator is
   * not found in the twiddles.
   *
   * @note This function assumes that the input data may have undergone reordering, and it adjusts the indices used for
   *       coset multiplication accordingly. The function handles both the cases where `twiddles` are used and where
   *       an `arbitrary_coset` is used.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::coset_mul(E* elements, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset)
  {
    uint64_t size = this->ntt_sub_logn.size;
    int batch_stride = this->config.columns_batch ? this->config.batch_size : 1;
    const bool needs_reorder_input = this->direction == NTTDir::kForward && (this->ntt_sub_logn.logn > HIERARCHY_1);

    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * size;

      for (uint64_t i = 1; i < size; ++i) {
        uint64_t idx = i;

        // Adjust the index if reorder logic was applied on the input
        if (needs_reorder_input) {
          int cur_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
          int next_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
          int subntt_idx = i >> cur_ntt_log_size;
          int element = i & ((1 << cur_ntt_log_size) - 1);
          idx = subntt_idx + (element << next_ntt_log_size);
        }

        // Apply coset multiplication based on the available coset information
        if (arbitrary_coset) {
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * arbitrary_coset[idx];
        } else {
          int twiddle_idx = coset_stride * idx;
          twiddle_idx = this->direction == NTTDir::kForward ? twiddle_idx : this->domain_max_size - twiddle_idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * this->twiddles[twiddle_idx];
        }
      }
    }
  }

  /**
   * @brief Handles the determination of the coset stride and potentially calculates an arbitrary coset.
   *
   * This function determines the appropriate coset stride based on the coset generator provided
   * in the configuration. If the coset generator is found within the precomputed twiddles, the
   * corresponding coset stride is returned. If the coset generator is not found, an arbitrary coset
   * is calculated dynamically and stored in the provided `arbitrary_coset` array.
   *
   * @param arbitrary_coset A unique pointer that will be allocated and filled with the arbitrary coset values
   *                        if the coset generator is not found in the precomputed twiddles.
   * @return int Returns the coset stride if found in the precomputed twiddles. If an arbitrary coset is calculated,
   *             returns 0 (since the stride is not applicable in this case).
   */

  template <typename S, typename E>
  int NttCpu<S, E>::find_or_generate_coset(std::unique_ptr<S[]>& arbitrary_coset)
  {
    int coset_stride = 0;

    if (config.coset_gen != S::one()) {
      try {
        coset_stride =
          CpuNttDomain<S>::s_ntt_domain.get_coset_stride(config.coset_gen); // Coset generator found in twiddles
      } catch (const std::out_of_range& oor) { // Coset generator not found in twiddles. Calculating arbitrary coset
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen =
          direction == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
        for (int i = 1; i <= domain_max_size; i++) {
          arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
        }
      }
    }

    return coset_stride;
  }

  /**
   * @brief Reorders elements between layers of hierarchy 1, based on sub-NTT structure.
   *
   * @param elements The array of elements to be reordered and refactored.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::hierarchy_1_reorder(E* elements)
  {
    const int sntt_size = 1 << this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
    const int nof_sntts = 1 << this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
    const int stride = this->config.columns_batch ? this->config.batch_size : 1;
    const uint64_t temp_elements_size = this->ntt_sub_logn.size * this->config.batch_size;

    auto temp_elements = std::make_unique<E[]>(temp_elements_size);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* cur_layer_output = this->config.columns_batch ? elements + batch : elements + batch * this->ntt_sub_logn.size;
      E* cur_temp_elements = this->config.columns_batch ? temp_elements.get() + batch
                                                        : temp_elements.get() + batch * this->ntt_sub_logn.size;
      for (int sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (int elem = 0; elem < sntt_size; elem++) {
          cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
            cur_layer_output[stride * (elem * nof_sntts + sntt_idx)];
        }
      }
    }
    std::copy(temp_elements.get(), temp_elements.get() + temp_elements_size, elements);
  }

  /**
   * @brief Reorders and optionally refactors the output data based on task coordinates and hierarchy level.
   *
   * This function reorders the output data based on the task coordinates so that the data will be in
   * the correct order for the next layer. If the function is dealing with a non-top-level
   * hierarchy and not limited to hierarchy 0, it will also apply a twiddle factor refactoring step before
   * moving on to the next hierarchy 1 layer.
   *
   * The reordering process involves reshuffling elements within the output array to match the required
   * structure, taking into account the sub-NTT sizes and indices.
   *
   * @param elements The array where the reordered and potentially refactored data will be stored.
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT computation hierarchy.
   * @param is_top_hirarchy A boolean indicating whether the function is operating at the top-level hierarchy (between
   * layers of hierarchy 1).
   *
   */
  template <typename S, typename E>
  eIcicleError
  NttCpu<S, E>::reorder_and_refactor_if_needed(E* elements, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy)
  {
    bool is_only_hierarchy_0 = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0] == 0;
    const bool refactor_pre_hierarchy_1_next_layer =
      (!is_only_hierarchy_0) && (!is_top_hirarchy) && (ntt_task_cordinates.hierarchy_1_layer_idx == 0);
    // const bool refactor_pre_hierarchy_1_next_layer = false;
    uint64_t size = (is_top_hirarchy || is_only_hierarchy_0)
                      ? this->ntt_sub_logn.size
                      : 1 << this->ntt_sub_logn.hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx];
    uint64_t temp_output_size = this->config.columns_batch ? size * this->config.batch_size : size;
    auto temp_output = std::make_unique<E[]>(temp_output_size);
    uint64_t idx = 0;
    uint64_t mem_idx = 0;
    uint64_t new_idx = 0;
    int subntt_idx;
    int element;
    int s0 = is_top_hirarchy
               ? this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0]
               : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0];
    int s1 = is_top_hirarchy
               ? this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1]
               : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
    int s2 = is_top_hirarchy
               ? this->ntt_sub_logn.hierarchy_1_layers_sub_logn[2]
               : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
    int p0, p1, p2;
    const int stride = this->config.columns_batch ? this->config.batch_size : 1;
    int rep = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t tw_idx = 0;
    E* hierarchy_1_subntt_output =
      elements +
      stride * (ntt_task_cordinates.hierarchy_1_subntt_idx
                << this->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                               // subntt_size
    for (int batch = 0; batch < rep; ++batch) {
      E* current_elements = this->config.columns_batch
                              ? hierarchy_1_subntt_output + batch
                              : hierarchy_1_subntt_output; // if columns_batch=false, then output is already shifted by
                                                           // batch*size when calling the function
      E* current_temp_output = this->config.columns_batch ? temp_output.get() + batch : temp_output.get();
      for (uint64_t i = 0; i < size; i++) {
        if (s2) {
          p0 = (i >> (s1 + s2));
          p1 = (((i >> s2) & ((1 << (s1)) - 1)) << s0);
          p2 = ((i & ((1 << s2) - 1)) << (s0 + s1));
          new_idx = p0 + p1 + p2;
        } else {
          subntt_idx = i >> s1;
          element = i & ((1 << s1) - 1);
          new_idx = subntt_idx + (element << s0);
        }
        if (refactor_pre_hierarchy_1_next_layer) {
          tw_idx = (this->direction == NTTDir::kForward)
                     ? ((this->domain_max_size >> this->ntt_sub_logn.logn) *
                        ntt_task_cordinates.hierarchy_1_subntt_idx * new_idx)
                     : this->domain_max_size - ((this->domain_max_size >> this->ntt_sub_logn.logn) *
                                                ntt_task_cordinates.hierarchy_1_subntt_idx * new_idx);
          current_temp_output[stride * new_idx] = current_elements[stride * i] * this->twiddles[tw_idx];
        } else {
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
    }
    std::copy(temp_output.get(), temp_output.get() + temp_output_size, hierarchy_1_subntt_output);
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Refactors the output of an hierarchy_0 sub-NTT after the NTT operation.
   *
   * This function refactors the output of an hierarchy_0 sub-NTT by applying twiddle factors to the elements
   * based on their indices. It prepares the data for further processing in subsequent layers of the NTT hierarchy.
   * Accesses corrected memory addresses, because reordering between layers of hierarchy 0 was skipped.
   *
   * @param elements The array of elements that have been transformed by the NTT.
   * @param ntt_task_cordinates The coordinates specifying the sub-NTT within the NTT hierarchy.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::refactor_output_hierarchy_0(E* elements, NttTaskCordinates ntt_task_cordinates)
  {
    int hierarchy_0_subntt_size =
      1 << NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                                 [ntt_task_cordinates.hierarchy_0_layer_idx];
    int hierarchy_0_nof_subntts =
      1 << NttCpu<S, E>::ntt_sub_logn
             .hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0]; // only relevant for layer 1
    int i, j, i_0;
    int ntt_size =
      ntt_task_cordinates.hierarchy_0_layer_idx == 0
        ? 1
            << (NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0] +
                NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1])
        : 1
            << (NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0] +
                NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1] +
                NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2]);
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t original_size = (1 << NttCpu<S, E>::ntt_sub_logn.logn);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* hierarchy_1_subntt_elements =
        elements +
        stride * (ntt_task_cordinates.hierarchy_1_subntt_idx
                  << NttCpu<S, E>::ntt_sub_logn
                       .hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
      E* elements_of_current_batch = this->config.columns_batch ? hierarchy_1_subntt_elements + batch
                                                                : hierarchy_1_subntt_elements + batch * original_size;
      for (int elem = 0; elem < hierarchy_0_subntt_size; elem++) {
        uint64_t elem_mem_idx = stride * idx_in_mem(ntt_task_cordinates, elem);
        i = (ntt_task_cordinates.hierarchy_0_layer_idx == 0)
              ? elem
              : elem * hierarchy_0_nof_subntts + ntt_task_cordinates.hierarchy_0_subntt_idx;
        j = (ntt_task_cordinates.hierarchy_0_layer_idx == 0) ? ntt_task_cordinates.hierarchy_0_subntt_idx
                                                             : ntt_task_cordinates.hierarchy_0_block_idx;
        uint64_t tw_idx = (this->direction == NTTDir::kForward)
                            ? ((this->domain_max_size / ntt_size) * j * i)
                            : this->domain_max_size - ((this->domain_max_size / ntt_size) * j * i);
        elements_of_current_batch[elem_mem_idx] = elements_of_current_batch[elem_mem_idx] * this->twiddles[tw_idx];
      }
    }
  }

  /**
   * @brief Pushes tasks for the hierarchy_1 hierarchy of the NTT computation.
   *
   * This function organizes and pushes tasks for the hierarchy_1 hierarchy level of the NTT computation
   * into the task manager. It iterates over the layers and sub-NTTs, scheduling the necessary
   * computations while ensuring that the data elements remain close in memory for efficiency,
   * assuming that in this hierarchy the sub-NTT fits in the cache.
   * If multiple layers are involved, Instead of sorting between layers, we are using the function
   * idx_in_mem to calculate the memory index of each element a final reorder task is pushed to fix
   * the order after processing.
   *
   * @param input The input array of elements to be processed.
   * @param ntt_task_cordinates The coordinates specifying the current sub-NTT within the NTT hierarchy.
   * @param ntt_tasks_manager The task manager responsible for handling the scheduling of tasks.
   * @return eIcicleError Returns `SUCCESS` if all tasks are successfully pushed to the task manager.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::hierarchy1_push_tasks(
    E* input, NttTaskCordinates ntt_task_cordinates, NttTasksManager<S, E>& ntt_tasks_manager)
  {
    uint64_t original_size = (this->ntt_sub_logn.size);
    int nof_hierarchy_0_layers =
      (this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2] != 0)   ? 3
      : (this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1] != 0) ? 2
                                                                                                            : 1;
    int log_nof_blocks;
    int log_nof_subntts;
    for (ntt_task_cordinates.hierarchy_0_layer_idx = 0;
         ntt_task_cordinates.hierarchy_0_layer_idx < nof_hierarchy_0_layers;
         ntt_task_cordinates.hierarchy_0_layer_idx++) {
      if (ntt_task_cordinates.hierarchy_0_layer_idx == 0) {
        log_nof_blocks = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
        log_nof_subntts = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
      } else if (ntt_task_cordinates.hierarchy_0_layer_idx == 1) {
        log_nof_blocks = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
        log_nof_subntts = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0];
      } else {
        log_nof_blocks = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0] +
                         this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
        log_nof_subntts = 0;
        ntt_task_cordinates.hierarchy_0_subntt_idx = 0; // not relevant for layer 2
      }
      for (ntt_task_cordinates.hierarchy_0_block_idx = 0;
           ntt_task_cordinates.hierarchy_0_block_idx < (1 << log_nof_blocks);
           ntt_task_cordinates.hierarchy_0_block_idx++) {
        for (ntt_task_cordinates.hierarchy_0_subntt_idx = 0;
             ntt_task_cordinates.hierarchy_0_subntt_idx < (1 << log_nof_subntts);
             ntt_task_cordinates.hierarchy_0_subntt_idx++) {
          ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
        }
      }
    }
    if (nof_hierarchy_0_layers > 1) { // all ntt tasks in hierarchy 1 are pushed, now push reorder task so that the data
                                      // is in the correct order for the next hierarchy 1 layer
      ntt_task_cordinates = {
        ntt_task_cordinates.hierarchy_1_layer_idx, ntt_task_cordinates.hierarchy_1_subntt_idx, nof_hierarchy_0_layers,
        0, 0};
      ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, true); // reorder=true
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Executes the NTT on a sub-NTT at the hierarchy_0 hierarchy level.
   *
   * This function applies the NTT on a sub-NTT specified by the task coordinates at the hierarchy_0 level.
   * hierarchy_0_dit_ntt transforming the data from the bit-reversed order (R) to natural order (N) so
   * this function first reorders the input elements by bit-reversing their indices, then performs the DIT NTT.
   * If further refactoring is required, the output is processed to prepare it for the next layer.
   *
   * @param input The input array of elements to be transformed.
   * @param ntt_task_cordinates The coordinates specifying the sub-NTT within the NTT hierarchy.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::hierarchy_0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates)
  {
    const uint64_t subntt_size =
      (1 << NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                                  [ntt_task_cordinates.hierarchy_0_layer_idx]);
    uint64_t original_size = (this->ntt_sub_logn.size);
    const uint64_t total_memory_size = original_size * this->config.batch_size;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    int offset = this->config.columns_batch ? this->config.batch_size : 1;
    E* current_input =
      input +
      offset * (ntt_task_cordinates.hierarchy_1_subntt_idx
                << NttCpu<S, E>::ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                               // subntt_size

    this->reorder_by_bit_reverse(ntt_task_cordinates, current_input, false); // R --> N

    // NTT/INTT
    this->hierarchy_0_dit_ntt(current_input, ntt_task_cordinates); // R --> N

    if (
      ntt_task_cordinates.hierarchy_0_layer_idx != 2 &&
      this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                    [ntt_task_cordinates.hierarchy_0_layer_idx + 1] != 0) {
      this->refactor_output_hierarchy_0(input, ntt_task_cordinates);
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Handles the execution and completion of tasks in the NTT computation.
   *
   * This function manages the lifecycle of tasks within the NTT computation at a given hierarchy level.
   * It retrieves available tasks, dispatches them for execution, and marks tasks as completed when done.
   * The function ensures that all tasks are processed, including handling idle states for tasks that
   * are waiting for dependencies to complete.
   *
   * @param tasks_manager The task manager responsible for managing the execution of individual tasks.
   * @param ntt_tasks_manager The manager responsible for handling the NTT tasks and their dependencies.
   * @param hierarchy_1_layer_idx The index of the current hierarchy_1 layer being processed.
   * @return eIcicleError Returns `SUCCESS` if all tasks are successfully handled.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::handle_pushed_tasks(
    TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, int hierarchy_1_layer_idx)
  {
    NttTask<S, E>* task_slot = nullptr;
    NttTaskParams<S, E> params;

    int nof_subntts_l1 = 1
                         << ((this->ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0]) +
                             (this->ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1]));
    while (ntt_tasks_manager.tasks_to_do()) {
      // There are tasks that are available or waiting

      if (ntt_tasks_manager.available_tasks()) {
        // Task is available to dispatch
        task_slot = tasks_manager->get_idle_or_completed_task();
        if (task_slot->is_completed()) { ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l1); }
        params = ntt_tasks_manager.get_available_task();
        task_slot->set_params(params);
        ntt_tasks_manager.erase_task_from_available_tasks_list();
        task_slot->dispatch();
      } else {
        // Wait for available tasks
        task_slot = tasks_manager->get_completed_task();
        ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l1);
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
    while (true) {
      task_slot = tasks_manager->get_completed_task(); // Get the last task (reorder task)
      if (task_slot == nullptr) {
        break;
      } else {
        task_slot->set_idle();
      }
    }
    return eIcicleError::SUCCESS;
  }

} // namespace ntt_cpu

namespace std {

  /**
   * @brief Hash function for NttTaskCordinates.
   *
   * Combines the hash values of the struct's members to allow
   * NttTaskCordinates to be used as a key in unordered containers.
   *
   * @param key The NttTaskCordinates to hash.
   * @return A size_t hash value.
   */
  template <>
  struct hash<ntt_cpu::NttTaskCordinates> {
    std::size_t operator()(const ntt_cpu::NttTaskCordinates& key) const
    {
      // Combine hash values of the members using a simple hash combiner
      return ((std::hash<int>()(key.hierarchy_1_layer_idx) ^ (std::hash<int>()(key.hierarchy_1_subntt_idx) << 1)) >>
              1) ^
             (std::hash<int>()(key.hierarchy_0_layer_idx) << 1) ^ (std::hash<int>()(key.hierarchy_0_block_idx) << 1) ^
             (std::hash<int>()(key.hierarchy_0_subntt_idx) << 1);
    }
  };
} // namespace std
