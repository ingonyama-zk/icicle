#pragma once
#include "icicle/backend/ntt_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/fields/field_config.h"
#include "icicle/vec_ops.h"
#include "tasks_manager.h"
#include "cpu_ntt_domain.h"

#include <_types/_uint32_t.h>
#include <sys/types.h>
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
    uint32_t hierarchy_1_layer_idx = 0;
    uint32_t hierarchy_1_subntt_idx = 0;
    uint32_t hierarchy_0_layer_idx = 0;
    uint32_t hierarchy_0_block_idx = 0;
    uint32_t hierarchy_0_subntt_idx = 0;

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
   * @method NttSubLogn(uint32_t logn) Initializes the struct based on the given `logn`.
   */
  struct NttSubLogn {
    uint32_t logn;                                                  // Original log_size of the problem
    uint64_t size;                                             // Original size of the problem
    std::vector<std::vector<uint32_t>> hierarchy_0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<uint32_t> hierarchy_1_layers_sub_logn;              // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubLogn(uint32_t logn) : logn(logn)
    {
      size = 1 << logn;
      if (logn > HIERARCHY_1) {
        // Initialize hierarchy_1_layers_sub_logn
        hierarchy_1_layers_sub_logn =
          std::vector<uint32_t>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn]));
        // Initialize hierarchy_0_layers_sub_logn
        hierarchy_0_layers_sub_logn = {
          std::vector<uint32_t>(
            std::begin(layers_sub_logn[hierarchy_1_layers_sub_logn[0]]),
            std::end(layers_sub_logn[hierarchy_1_layers_sub_logn[0]])),
          std::vector<uint32_t>(
            std::begin(layers_sub_logn[hierarchy_1_layers_sub_logn[1]]),
            std::end(layers_sub_logn[hierarchy_1_layers_sub_logn[1]]))};
      } else {
        hierarchy_1_layers_sub_logn = {0, 0, 0};
        hierarchy_0_layers_sub_logn = {
          std::vector<uint32_t>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn])), {0, 0, 0}};
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
    uint32_t domain_max_size;
    const S* twiddles;
    const S* winograd8_twiddles;
    const S* winograd16_twiddles;
    const S* winograd32_twiddles;

    NttCpu(uint32_t logn, NTTDir direction, const NTTConfig<S>& config)
        : ntt_sub_logn(logn), direction(direction), config(config)
    {
      domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size(); 
      twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
      winograd8_twiddles = direction == NTTDir::kForward ? CpuNttDomain<S>::s_ntt_domain.get_winograd8_twiddles() : CpuNttDomain<S>::s_ntt_domain.get_winograd8_twiddles_inv();
      winograd16_twiddles = direction == NTTDir::kForward ? CpuNttDomain<S>::s_ntt_domain.get_winograd16_twiddles() : CpuNttDomain<S>::s_ntt_domain.get_winograd16_twiddles_inv();
      winograd32_twiddles = direction == NTTDir::kForward ? CpuNttDomain<S>::s_ntt_domain.get_winograd32_twiddles() : CpuNttDomain<S>::s_ntt_domain.get_winograd32_twiddles_inv();
    }

    void reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy);
    eIcicleError copy_and_reorder_if_needed(const E* input, E* output);
    void ntt8win(E* elements, NttTaskCordinates ntt_task_cordinates);
    void ntt16win(E* elements, NttTaskCordinates ntt_task_cordinates);
    void ntt32win(E* elements, NttTaskCordinates ntt_task_cordinates);
    void hierarchy_0_dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates);
    void hierarchy_0_dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates);
    void coset_mul(E* elements, uint32_t coset_stride, const std::unique_ptr<S[]>& arbitrary_coset);
    uint32_t find_or_generate_coset(std::unique_ptr<S[]>& arbitrary_coset);
    void hierarchy_1_reorder(E* elements);
    eIcicleError
    reorder_and_refactor_if_needed(E* elements, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy);
    eIcicleError
    hierarchy1_push_tasks(E* input, NttTaskCordinates ntt_task_cordinates, NttTasksManager<S, E>& ntt_tasks_manager);
    eIcicleError hierarchy_0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates);
    eIcicleError handle_pushed_tasks(
      TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, uint32_t hierarchy_1_layer_idx);

  private:
    uint64_t bit_reverse(uint64_t n, uint32_t logn);
    uint64_t idx_in_mem(NttTaskCordinates ntt_task_cordinates, uint32_t element);
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
    TasksDependenciesCounters(NttSubLogn ntt_sub_logn, uint32_t hierarchy_1_layer_idx);

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
          for (uint32_t b = 0; b < ntt_cpu->config.batch_size; b++) {
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
    NttTasksManager(uint32_t logn);

    // Add a new task to the ntt_task_manager
    eIcicleError push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder);

    // Set a task as completed and update dependencies
    eIcicleError set_task_as_completed(NttTask<S, E>& completed_task, uint32_t nof_subntts_l1);

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
  TasksDependenciesCounters::TasksDependenciesCounters(NttSubLogn ntt_sub_logn, uint32_t hierarchy_1_layer_idx)
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
  NttTasksManager<S, E>::NttTasksManager(uint32_t logn)
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
  eIcicleError NttTasksManager<S, E>::set_task_as_completed(NttTask<S, E>& completed_task, uint32_t nof_subntts_l1)
  {
    ntt_cpu::NttTaskCordinates task_c = completed_task.get_coordinates();
    uint32_t nof_hierarchy_0_layers = counters[task_c.hierarchy_1_layer_idx].get_nof_hierarchy_0_layers();
    // Update dependencies in counters
    if (counters[task_c.hierarchy_1_layer_idx].decrement_counter(task_c)) {
      if (task_c.hierarchy_0_layer_idx < nof_hierarchy_0_layers - 1) {
        uint32_t dependent_subntt_count =
          (task_c.hierarchy_0_layer_idx == nof_hierarchy_0_layers - 1)
            ? 1
            : counters[task_c.hierarchy_1_layer_idx].get_dependent_subntt_count(task_c.hierarchy_0_layer_idx + 1);
        uint32_t stride = nof_subntts_l1 / dependent_subntt_count;
        for (uint32_t i = 0; i < dependent_subntt_count; i++) {
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
  uint64_t NttCpu<S, E>::bit_reverse(uint64_t n, uint32_t logn)
  {
    uint32_t rev = 0;
    for (uint32_t j = 0; j < logn; ++j) {
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
  uint64_t NttCpu<S, E>::idx_in_mem(NttTaskCordinates ntt_task_cordinates, uint32_t element_idx)
  {
    uint32_t s0 = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0];
    uint32_t s1 = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
    uint32_t s2 = this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
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
    uint32_t subntt_log_size = is_top_hirarchy
                            ? (this->ntt_sub_logn.logn)
                            : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                                            [ntt_task_cordinates.hierarchy_0_layer_idx];
    uint64_t original_size = (1 << this->ntt_sub_logn.logn);
    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
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
    const uint32_t stride = config.columns_batch ? config.batch_size : 1;
    const uint32_t logn = static_cast<uint32_t>(std::log2(this->ntt_sub_logn.size));
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
      uint32_t cur_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
      uint32_t next_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];

      for (uint32_t batch = 0; batch < config.batch_size; ++batch) {
        const E* input_batch = config.columns_batch ? (input + batch) : (input + batch * this->ntt_sub_logn.size);
        E* output_batch =
          config.columns_batch ? (temp_output + batch) : (temp_output + batch * this->ntt_sub_logn.size);

        for (uint64_t i = 0; i < this->ntt_sub_logn.size; ++i) {
          uint32_t subntt_idx = i >> cur_ntt_log_size;
          uint32_t element = i & ((1 << cur_ntt_log_size) - 1);
          uint64_t new_idx = bit_rev ? bit_reverse(subntt_idx + (element << next_ntt_log_size), logn)
                                     : subntt_idx + (element << next_ntt_log_size);
          output_batch[stride * i] = input_batch[stride * new_idx];
        }
      }

    } else if (bit_rev) {
      // Only bit-reverse reordering needed
      for (uint32_t batch = 0; batch < config.batch_size; ++batch) {
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
  void NttCpu<S, E>::ntt8win(E* elements, NttTaskCordinates ntt_task_cordinates) // N --> N
  {
    // std::cout << "ntt8win" << std::endl;
    // bool inv = (this->direction == NTTDir::kInverse);
    E T;
    std::vector<uint32_t> index_in_mem(8);
    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (uint32_t i = 0; i < 8; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_cordinates, i);
    }
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);
      
      T = current_elements[index_in_mem[3]] - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[3]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[1]] - current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[2]] + current_elements[index_in_mem[6]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[2]] - current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[4]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[4]];

      current_elements[index_in_mem[2]] = current_elements[index_in_mem[2]] * this->winograd8_twiddles[0];

      current_elements[index_in_mem[4]] = current_elements[index_in_mem[6]] + current_elements[index_in_mem[1]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[6]] - current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[3]] + T;
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[3]] - T;
      T = current_elements[index_in_mem[5]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[5]] - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[2]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[2]];

      current_elements[index_in_mem[1]] = current_elements[index_in_mem[1]] * this->winograd8_twiddles[1];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[5]] * this->winograd8_twiddles[0];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[3]] * this->winograd8_twiddles[2];

      current_elements[index_in_mem[2]] = current_elements[index_in_mem[6]] + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[6]] - current_elements[index_in_mem[5]];

      current_elements[index_in_mem[5]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[3]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[1]] - current_elements[index_in_mem[3]];

      current_elements[index_in_mem[1]] = current_elements[index_in_mem[7]] + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[7]] - current_elements[index_in_mem[5]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[3]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[3]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[4]] + T;
      current_elements[index_in_mem[4]] = current_elements[index_in_mem[4]] - T;
    }
  }


  template <typename S, typename E>
  void NttCpu<S, E>::ntt16win(E* elements, NttTaskCordinates ntt_task_cordinates) // N --> N
  {
    // std::cout << "ntt16win" << std::endl;
    E T;
    std::vector<uint32_t> index_in_mem(16);
    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (uint32_t i = 0; i < 16; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_cordinates, i);
    }
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);


      T = current_elements[index_in_mem[0]] + current_elements[index_in_mem[8]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[8]];
      current_elements[index_in_mem[8]] = current_elements[index_in_mem[4]] + current_elements[index_in_mem[12]];
      current_elements[index_in_mem[4]]  = current_elements[index_in_mem[4]] - current_elements[index_in_mem[12]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[2]] + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[2]]  = current_elements[index_in_mem[2]] - current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = current_elements[index_in_mem[6]] + current_elements[index_in_mem[14]];
      current_elements[index_in_mem[6]]  = current_elements[index_in_mem[6]] - current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[9]];
      current_elements[index_in_mem[1]]  = current_elements[index_in_mem[1]] - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]]  = current_elements[index_in_mem[5]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[5]]  = current_elements[index_in_mem[5]] - current_elements[index_in_mem[13]];
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[3]] + current_elements[index_in_mem[11]];
      current_elements[index_in_mem[3]]  = current_elements[index_in_mem[3]] - current_elements[index_in_mem[11]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[7]] + current_elements[index_in_mem[15]];
      current_elements[index_in_mem[7]]  = current_elements[index_in_mem[7]] - current_elements[index_in_mem[15]];
      current_elements[index_in_mem[4]] = this->winograd16_twiddles[3] * current_elements[index_in_mem[4]];

      // 2
      current_elements[index_in_mem[15]] = T  + current_elements[index_in_mem[8]];
      T  = T  - current_elements[index_in_mem[8]];
      current_elements[index_in_mem[8]]  = current_elements[index_in_mem[0]]  + current_elements[index_in_mem[4]];
      current_elements[index_in_mem[0]]  = current_elements[index_in_mem[0]]  - current_elements[index_in_mem[4]];
      current_elements[index_in_mem[4]]  = current_elements[index_in_mem[12]] + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[12]] - current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = current_elements[index_in_mem[2]]  + current_elements[index_in_mem[6]];
      current_elements[index_in_mem[2]]  = current_elements[index_in_mem[2]]  - current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]]  = current_elements[index_in_mem[14]] + current_elements[index_in_mem[9]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[14]] - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]]  = current_elements[index_in_mem[13]] + current_elements[index_in_mem[11]];
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[13]] - current_elements[index_in_mem[11]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[1]]  + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[1]]  = current_elements[index_in_mem[1]]  - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]]  = current_elements[index_in_mem[3]]  + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[3]]  = current_elements[index_in_mem[3]]  - current_elements[index_in_mem[5]];

      current_elements[index_in_mem[12]] = this->winograd16_twiddles[5] * current_elements[index_in_mem[12]];
      current_elements[index_in_mem[10]] = this->winograd16_twiddles[6] * current_elements[index_in_mem[10]];
      current_elements[index_in_mem[2]]  = this->winograd16_twiddles[7] * current_elements[index_in_mem[2]];

      // 3
      current_elements[index_in_mem[5]]  = current_elements[index_in_mem[10]] + current_elements[index_in_mem[2]];
      current_elements[index_in_mem[10]] = current_elements[index_in_mem[10]] - current_elements[index_in_mem[2]];
      current_elements[index_in_mem[2]]  = current_elements[index_in_mem[6]]  + current_elements[index_in_mem[9]];
      current_elements[index_in_mem[6]]  = current_elements[index_in_mem[6]]  - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]]  = current_elements[index_in_mem[14]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[14]] - current_elements[index_in_mem[13]];

      current_elements[index_in_mem[13]] = current_elements[index_in_mem[11]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[13]] = this->winograd16_twiddles[14] * current_elements[index_in_mem[13]];
      current_elements[index_in_mem[11]] = this->winograd16_twiddles[12] * current_elements[index_in_mem[11]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[7]]  = this->winograd16_twiddles[13] * current_elements[index_in_mem[7]]  + current_elements[index_in_mem[13]];

      current_elements[index_in_mem[13]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[3]];
      current_elements[index_in_mem[13]] = this->winograd16_twiddles[17] * current_elements[index_in_mem[13]];
      current_elements[index_in_mem[1]]  = this->winograd16_twiddles[15] * current_elements[index_in_mem[1]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[3]]  = this->winograd16_twiddles[16] * current_elements[index_in_mem[3]] + current_elements[index_in_mem[13]];

      // 4
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[15]] + current_elements[index_in_mem[4]];
      current_elements[index_in_mem[15]] = current_elements[index_in_mem[15]] - current_elements[index_in_mem[4]];
      current_elements[index_in_mem[4]]  = T  + current_elements[index_in_mem[12]];
      T  = T  - current_elements[index_in_mem[12]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[8]]  + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[8]]  = current_elements[index_in_mem[8]]  - current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]]  = current_elements[index_in_mem[0]]  + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[0]]  = current_elements[index_in_mem[0]]  - current_elements[index_in_mem[10]];

      current_elements[index_in_mem[6]]   = this->winograd16_twiddles[9]  * current_elements[index_in_mem[6]];
      current_elements[index_in_mem[9]]   = this->winograd16_twiddles[10] * current_elements[index_in_mem[9]];
      current_elements[index_in_mem[14]]  = this->winograd16_twiddles[11] * current_elements[index_in_mem[14]];

      current_elements[index_in_mem[10]] = current_elements[index_in_mem[9]]  + current_elements[index_in_mem[14]];
      current_elements[index_in_mem[9]]  = current_elements[index_in_mem[9]]  - current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[11]] + current_elements[index_in_mem[1]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[11]] - current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]]  = current_elements[index_in_mem[7]]  + current_elements[index_in_mem[3]];
      current_elements[index_in_mem[7]]  = current_elements[index_in_mem[7]]  - current_elements[index_in_mem[3]];

      // 5
      current_elements[index_in_mem[3]]  = current_elements[index_in_mem[13]] + current_elements[index_in_mem[2]];
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[13]] - current_elements[index_in_mem[2]];
      current_elements[index_in_mem[2]]  = current_elements[index_in_mem[15]] + current_elements[index_in_mem[6]];
      current_elements[index_in_mem[15]] = current_elements[index_in_mem[15]] - current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]]  = current_elements[index_in_mem[4]]  + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[4]]  = current_elements[index_in_mem[4]]  - current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = T + current_elements[index_in_mem[9]];
      T  = T - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]]  = current_elements[index_in_mem[12]] + current_elements[index_in_mem[14]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[12]] - current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[8]]  + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[8]]  = current_elements[index_in_mem[8]]  - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]]  = current_elements[index_in_mem[5]]  + current_elements[index_in_mem[1]];
      current_elements[index_in_mem[5]]  = current_elements[index_in_mem[5]]  - current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]]  = current_elements[index_in_mem[0]]  + current_elements[index_in_mem[11]];
      current_elements[index_in_mem[0]]  = current_elements[index_in_mem[0]]  - current_elements[index_in_mem[11]];

      //reorder + return
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[0]];
      current_elements[index_in_mem[0]]  = current_elements[index_in_mem[3]];
      current_elements[index_in_mem[3]]  = current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]]  = current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]]  = current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]]  = current_elements[index_in_mem[12]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[15]];
      current_elements[index_in_mem[15]] = current_elements[index_in_mem[11]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]]  = current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = T;
      T  = current_elements[index_in_mem[8]];
      current_elements[index_in_mem[8]]  = current_elements[index_in_mem[13]];
      current_elements[index_in_mem[13]] = T;
      T  = current_elements[index_in_mem[4]];
      current_elements[index_in_mem[4]]  = current_elements[index_in_mem[2]];
      current_elements[index_in_mem[2]]  = current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]]  = current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = T;
    }
  }

    template <typename S, typename E>
  void NttCpu<S, E>::ntt32win(E* elements, NttTaskCordinates ntt_task_cordinates) // N --> N
  {
    // std::cout << "ntt32win" << std::endl;
    bool inv = (this->direction == NTTDir::kInverse);
    std::vector<E> temp_0(46);
    std::vector<E> temp_1(46);
    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    std::vector<uint32_t> index_in_mem(32);
    for (uint32_t i = 0; i < 32; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_cordinates, i);
    }

    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);

      /*  Stage s00  */
      temp_0[0] = current_elements[index_in_mem[0]];
      temp_0[1] = current_elements[index_in_mem[2]];
      temp_0[2] = current_elements[index_in_mem[4]];
      temp_0[3] = current_elements[index_in_mem[6]];
      temp_0[4] = current_elements[index_in_mem[8]];
      temp_0[5] = current_elements[index_in_mem[10]];
      temp_0[6] = current_elements[index_in_mem[12]];
      temp_0[7] = current_elements[index_in_mem[14]];
      temp_0[8] = current_elements[index_in_mem[16]];
      temp_0[9] = current_elements[index_in_mem[18]];
      temp_0[10] = current_elements[index_in_mem[20]];
      temp_0[11] = current_elements[index_in_mem[22]];
      temp_0[12] = current_elements[index_in_mem[24]];
      temp_0[13] = current_elements[index_in_mem[26]];
      temp_0[14] = current_elements[index_in_mem[28]];
      temp_0[15] = current_elements[index_in_mem[30]];
      temp_0[16] = current_elements[index_in_mem[1]];
      temp_0[17] = current_elements[index_in_mem[3]];
      temp_0[18] = current_elements[index_in_mem[5]];
      temp_0[19] = current_elements[index_in_mem[7]];
      temp_0[20] = current_elements[index_in_mem[9]];
      temp_0[21] = current_elements[index_in_mem[11]];
      temp_0[22] = current_elements[index_in_mem[13]];
      temp_0[23] = current_elements[index_in_mem[15]];
      temp_0[24] = current_elements[index_in_mem[17]];
      temp_0[25] = current_elements[index_in_mem[19]];
      temp_0[26] = current_elements[index_in_mem[21]];
      temp_0[27] = current_elements[index_in_mem[23]];
      temp_0[28] = current_elements[index_in_mem[25]];
      temp_0[29] = current_elements[index_in_mem[27]];
      temp_0[30] = current_elements[index_in_mem[29]];
      temp_0[31] = current_elements[index_in_mem[31]];

      /*  Stage s01  */

      temp_1[0] = temp_0[0];
      temp_1[1] = temp_0[2];
      temp_1[2] = temp_0[4];
      temp_1[3] = temp_0[6];
      temp_1[4] = temp_0[8];
      temp_1[5] = temp_0[10];
      temp_1[6] = temp_0[12];
      temp_1[7] = temp_0[14];
      temp_1[8] = temp_0[1];
      temp_1[9] = temp_0[3];
      temp_1[10] = temp_0[5];
      temp_1[11] = temp_0[7];
      temp_1[12] = temp_0[9];
      temp_1[13] = temp_0[11];
      temp_1[14] = temp_0[13];
      temp_1[15] = temp_0[15];
      temp_1[16] = temp_0[16] + temp_0[24];
      temp_1[17] = temp_0[17] + temp_0[25];
      temp_1[18] = temp_0[18] + temp_0[26];
      temp_1[19] = temp_0[19] + temp_0[27];
      temp_1[20] = temp_0[20] + temp_0[28];
      temp_1[21] = temp_0[21] + temp_0[29];
      temp_1[22] = temp_0[22] + temp_0[30];
      temp_1[23] = temp_0[23] + temp_0[31];
      temp_1[24] = temp_0[16] - temp_0[24];
      temp_1[25] = temp_0[17] - temp_0[25];
      temp_1[26] = temp_0[18] - temp_0[26];
      temp_1[27] = temp_0[19] - temp_0[27];
      temp_1[28] = temp_0[20] - temp_0[28];
      temp_1[29] = temp_0[21] - temp_0[29];
      temp_1[30] = temp_0[22] - temp_0[30];
      temp_1[31] = temp_0[23] - temp_0[31];

      /*  Stage s02  */

      temp_0[0] = temp_1[0];
      temp_0[1] = temp_1[2];
      temp_0[2] = temp_1[4];
      temp_0[3] = temp_1[6];
      temp_0[4] = temp_1[1];
      temp_0[5] = temp_1[3];
      temp_0[6] = temp_1[5];
      temp_0[7] = temp_1[7];
      temp_0[8] = temp_1[8];
      temp_0[9] = temp_1[9];
      temp_0[10] = temp_1[10];
      temp_0[11] = temp_1[11];
      temp_0[12] = temp_1[12];
      temp_0[13] = temp_1[13];
      temp_0[14] = temp_1[14];
      temp_0[15] = temp_1[15];
      temp_0[16] = temp_1[16];
      temp_0[17] = temp_1[17];
      temp_0[18] = temp_1[18];
      temp_0[19] = temp_1[19];
      temp_0[20] = temp_1[20];
      temp_0[21] = temp_1[21];
      temp_0[22] = temp_1[22];
      temp_0[23] = temp_1[23];
      temp_0[24] = temp_1[24];
      temp_0[25] = temp_1[25];
      temp_0[26] = temp_1[26];
      temp_0[27] = temp_1[27];
      temp_0[28] = temp_1[31];
      temp_0[29] = temp_1[30];
      temp_0[30] = temp_1[29];
      temp_0[31] = temp_1[28];

      /*  Stage s03  */

      temp_1[0] = temp_0[0];
      temp_1[1] = temp_0[1];
      temp_1[2] = temp_0[2];
      temp_1[3] = temp_0[3];
      temp_1[4] = temp_0[4];
      temp_1[5] = temp_0[5];
      temp_1[6] = temp_0[6];
      temp_1[7] = temp_0[7];
      temp_1[8] = temp_0[12] + temp_0[8];
      temp_1[9] = temp_0[13] + temp_0[9];
      temp_1[10] = temp_0[10] + temp_0[14];
      temp_1[11] = temp_0[11] + temp_0[15];
      temp_1[12] = temp_0[8] - temp_0[12];
      temp_1[13] = temp_0[9] - temp_0[13];
      temp_1[14] = temp_0[10] - temp_0[14];
      temp_1[15] = temp_0[11] - temp_0[15];
      temp_1[16] = temp_0[16] + temp_0[20];
      temp_1[17] = temp_0[17] + temp_0[21];
      temp_1[18] = temp_0[18] + temp_0[22];
      temp_1[19] = temp_0[19] + temp_0[23];
      temp_1[20] = temp_0[16] - temp_0[20];
      temp_1[21] = temp_0[17] - temp_0[21];
      temp_1[22] = temp_0[18] - temp_0[22];
      temp_1[23] = temp_0[19] - temp_0[23];
      temp_1[24] = temp_0[24] + temp_0[28];
      temp_1[25] = temp_0[25] + temp_0[29];
      temp_1[26] = temp_0[26] + temp_0[30];
      temp_1[27] = temp_0[27] + temp_0[31];
      temp_1[28] = temp_0[24] - temp_0[28];
      temp_1[29] = temp_0[25] - temp_0[29];
      temp_1[30] = temp_0[26] - temp_0[30];
      temp_1[31] = temp_0[27] - temp_0[31];

      /*  Stage s04  */

      temp_0[0] = temp_1[0];
      temp_0[1] = temp_1[2];
      temp_0[2] = temp_1[1];
      temp_0[3] = temp_1[3];
      temp_0[4] = temp_1[4];
      temp_0[5] = temp_1[5];
      temp_0[6] = temp_1[6];
      temp_0[7] = temp_1[7];
      temp_0[8] = temp_1[8];
      temp_0[9] = temp_1[9];
      temp_0[10] = temp_1[10];
      temp_0[11] = temp_1[11];
      temp_0[12] = temp_1[12];
      temp_0[13] = temp_1[13];
      temp_0[14] = temp_1[15];
      temp_0[15] = temp_1[14];
      temp_0[16] = temp_1[16];
      temp_0[17] = temp_1[17];
      temp_0[18] = temp_1[18];
      temp_0[19] = temp_1[19];
      temp_0[20] = temp_1[20];
      temp_0[21] = temp_1[21];
      temp_0[22] = temp_1[23];
      temp_0[23] = temp_1[22];
      temp_0[24] = temp_1[24];
      temp_0[25] = temp_1[27];
      temp_0[26] = temp_1[26];
      temp_0[27] = temp_1[25];
      temp_0[28] = temp_1[28];
      temp_0[29] = temp_1[31];
      temp_0[30] = temp_1[30];
      temp_0[31] = temp_1[29];

      /*  Stage s05  */

      temp_1[0] = temp_0[0];
      temp_1[1] = temp_0[1];
      temp_1[2] = temp_0[2];
      temp_1[3] = temp_0[3];
      temp_1[4] = temp_0[4] + temp_0[6];
      temp_1[5] = temp_0[5] + temp_0[7];
      temp_1[6] = temp_0[4] - temp_0[6];
      temp_1[7] = temp_0[5] - temp_0[7];
      temp_1[8] = temp_0[10] + temp_0[8];
      temp_1[9] = temp_0[11] + temp_0[9];
      temp_1[10] = temp_0[8] - temp_0[10];
      temp_1[11] = temp_0[9] - temp_0[11];
      temp_1[12] = temp_0[12] + temp_0[14];
      temp_1[13] = temp_0[13] + temp_0[15];
      temp_1[14] = temp_0[12] - temp_0[14];
      temp_1[15] = temp_0[13] - temp_0[15];
      temp_1[16] = temp_0[16] + temp_0[18];
      temp_1[17] = temp_0[17] + temp_0[19];
      temp_1[18] = temp_0[16] - temp_0[18];
      temp_1[19] = temp_0[17] - temp_0[19];
      temp_1[20] = temp_0[20] + temp_0[22];
      temp_1[21] = temp_0[21] + temp_0[23];
      temp_1[22] = temp_0[20] - temp_0[22];
      temp_1[23] = temp_0[21] - temp_0[23];
      temp_1[24] = temp_0[24];
      temp_1[25] = temp_0[25];
      temp_1[26] = temp_0[26];
      temp_1[27] = temp_0[27];
      temp_1[28] = temp_0[24] + temp_0[26];
      temp_1[29] = temp_0[25] + temp_0[27];
      temp_1[30] = temp_0[28];
      temp_1[31] = temp_0[29];
      temp_1[32] = temp_0[30];
      temp_1[33] = temp_0[31];
      temp_1[34] = temp_0[28] + temp_0[30];
      temp_1[35] = temp_0[29] + temp_0[31];

      /*  Stage s06  */

      temp_0[0] = temp_1[0] + temp_1[1];
      temp_0[1] = temp_1[0] - temp_1[1];
      temp_0[2] = temp_1[2] + temp_1[3];
      temp_0[3] = temp_1[2] - temp_1[3];
      temp_0[4] = temp_1[4] + temp_1[5];
      temp_0[5] = temp_1[4] - temp_1[5];
      temp_0[6] = temp_1[6] + temp_1[7];
      temp_0[7] = temp_1[6] - temp_1[7];
      temp_0[8] = temp_1[8] + temp_1[9];
      temp_0[9] = temp_1[8] - temp_1[9];
      temp_0[10] = temp_1[10] + temp_1[11];
      temp_0[11] = temp_1[10] - temp_1[11];
      temp_0[12] = temp_1[12];
      temp_0[13] = temp_1[13];
      temp_0[14] = temp_1[14];
      temp_0[15] = temp_1[15];
      temp_0[16] = temp_1[16] + temp_1[17];
      temp_0[17] = temp_1[16] - temp_1[17];
      temp_0[18] = temp_1[18] + temp_1[19];
      temp_0[19] = temp_1[18] - temp_1[19];
      temp_0[20] = temp_1[20];
      temp_0[21] = temp_1[21];
      temp_0[22] = temp_1[22];
      temp_0[23] = temp_1[23];
      temp_0[24] = temp_1[24];
      temp_0[25] = temp_1[25];
      temp_0[26] = temp_1[26];
      temp_0[27] = temp_1[27];
      temp_0[28] = temp_1[28];
      temp_0[29] = temp_1[29];
      temp_0[30] = temp_1[30];
      temp_0[31] = temp_1[31];
      temp_0[32] = temp_1[32];
      temp_0[33] = temp_1[33];
      temp_0[34] = temp_1[34];
      temp_0[35] = temp_1[35];

      /*  Stage s07  */

      temp_1[0] = temp_0[0];
      temp_1[1] = temp_0[1];
      temp_1[2] = temp_0[2];
      temp_1[3] = temp_0[3];
      temp_1[4] = temp_0[4];
      temp_1[5] = temp_0[5];
      temp_1[6] = temp_0[6];
      temp_1[7] = temp_0[7];
      temp_1[8] = temp_0[8];
      temp_1[9] = temp_0[9];
      temp_1[10] = temp_0[10];
      temp_1[11] = temp_0[11];
      temp_1[12] = temp_0[12];
      temp_1[13] = temp_0[13];
      temp_1[14] = temp_0[12] + temp_0[13];
      temp_1[15] = temp_0[14];
      temp_1[16] = temp_0[15];
      temp_1[17] = temp_0[14] + temp_0[15];
      temp_1[18] = temp_0[16];
      temp_1[19] = temp_0[17];
      temp_1[20] = temp_0[18];
      temp_1[21] = temp_0[19];
      temp_1[22] = temp_0[20];
      temp_1[23] = temp_0[21];
      temp_1[24] = temp_0[20] + temp_0[21];
      temp_1[25] = temp_0[22];
      temp_1[26] = temp_0[23];
      temp_1[27] = temp_0[22] + temp_0[23];
      temp_1[28] = temp_0[24];
      temp_1[29] = temp_0[25];
      temp_1[30] = temp_0[24] + temp_0[25];
      temp_1[31] = temp_0[26];
      temp_1[32] = temp_0[27];
      temp_1[33] = temp_0[26] + temp_0[27];
      temp_1[34] = temp_0[28];
      temp_1[35] = temp_0[29];
      temp_1[36] = temp_0[28] + temp_0[29];
      temp_1[37] = temp_0[30];
      temp_1[38] = temp_0[31];
      temp_1[39] = temp_0[30] + temp_0[31];
      temp_1[40] = temp_0[32];
      temp_1[41] = temp_0[33];
      temp_1[42] = temp_0[32] + temp_0[33];
      temp_1[43] = temp_0[34];
      temp_1[44] = temp_0[35];
      temp_1[45] = temp_0[34] + temp_0[35];

      /*  Stage s08  */

      for (uint32_t i = 0; i < 46; i++) {
        temp_0[i] = temp_1[i] * this->winograd32_twiddles[i];
      }

      /*  Stage s09  */

      temp_1[0] = temp_0[0];
      temp_1[1] = temp_0[1];
      temp_1[2] = temp_0[2];
      temp_1[3] = temp_0[3];
      temp_1[4] = temp_0[4];
      temp_1[5] = temp_0[5];
      temp_1[6] = temp_0[6];
      temp_1[7] = temp_0[7];
      temp_1[8] = temp_0[8];
      temp_1[9] = temp_0[9];
      temp_1[10] = temp_0[10];
      temp_1[11] = temp_0[11];
      temp_1[12] = temp_0[12] + temp_0[14];
      temp_1[13] = temp_0[13] + temp_0[14];
      temp_1[14] = temp_0[15] + temp_0[17];
      temp_1[15] = temp_0[16] + temp_0[17];
      temp_1[16] = temp_0[18];
      temp_1[17] = temp_0[19];
      temp_1[18] = temp_0[20];
      temp_1[19] = temp_0[21];
      temp_1[20] = temp_0[22] + temp_0[24];
      temp_1[21] = temp_0[23] + temp_0[24];
      temp_1[22] = temp_0[25] + temp_0[27];
      temp_1[23] = temp_0[26] + temp_0[27];
      temp_1[24] = temp_0[28] + temp_0[30];
      temp_1[25] = temp_0[29] + temp_0[30];
      temp_1[26] = temp_0[31] + temp_0[33];
      temp_1[27] = temp_0[32] + temp_0[33];
      temp_1[28] = temp_0[34] + temp_0[36];
      temp_1[29] = temp_0[35] + temp_0[36];
      temp_1[30] = temp_0[37] + temp_0[39];
      temp_1[31] = temp_0[38] + temp_0[39];
      temp_1[32] = temp_0[40] + temp_0[42];
      temp_1[33] = temp_0[41] + temp_0[42];
      temp_1[34] = temp_0[43] + temp_0[45];
      temp_1[35] = temp_0[44] + temp_0[45];

      /*  Stage s10  */

      temp_0[0] = temp_1[0];
      temp_0[1] = temp_1[1];
      temp_0[2] = temp_1[2];
      temp_0[3] = temp_1[3];
      temp_0[4] = temp_1[4];
      temp_0[5] = temp_1[5];
      temp_0[6] = temp_1[6] + temp_1[7];
      temp_0[7] = temp_1[6] - temp_1[7];
      temp_0[8] = temp_1[8];
      temp_0[9] = temp_1[9];
      temp_0[10] = temp_1[10] + temp_1[11];
      temp_0[11] = temp_1[10] - temp_1[11];
      temp_0[12] = temp_1[12];
      temp_0[13] = temp_1[13];
      temp_0[14] = temp_1[14];
      temp_0[15] = temp_1[15];
      temp_0[16] = temp_1[16];
      temp_0[17] = temp_1[17];
      temp_0[18] = temp_1[18] + temp_1[19];
      temp_0[19] = temp_1[18] - temp_1[19];
      temp_0[20] = temp_1[20];
      temp_0[21] = temp_1[21];
      temp_0[22] = temp_1[22];
      temp_0[23] = temp_1[23];
      temp_0[24] = temp_1[24];
      temp_0[25] = temp_1[25];
      temp_0[26] = temp_1[26];
      temp_0[27] = temp_1[27];
      temp_0[28] = temp_1[28];
      temp_0[29] = temp_1[29];
      temp_0[30] = temp_1[30];
      temp_0[31] = temp_1[31];
      temp_0[32] = temp_1[32];
      temp_0[33] = temp_1[33];
      temp_0[34] = temp_1[34];
      temp_0[35] = temp_1[35];

      /*  Stage s11  */

      temp_1[0] = temp_0[0] + temp_0[2];
      temp_1[1] = temp_0[1] + temp_0[3];
      temp_1[2] = temp_0[0] - temp_0[2];
      temp_1[3] = temp_0[1] - temp_0[3];
      temp_1[4] = temp_0[4];
      temp_1[5] = temp_0[5];
      temp_1[6] = temp_0[6];
      temp_1[7] = temp_0[7];
      temp_1[8] = temp_0[8];
      temp_1[9] = temp_0[9];
      temp_1[10] = temp_0[10];
      temp_1[11] = temp_0[11];
      temp_1[12] = temp_0[12] + temp_0[14];
      temp_1[13] = temp_0[13] + temp_0[15];
      temp_1[14] = temp_0[12] - temp_0[14];
      temp_1[15] = temp_0[13] - temp_0[15];
      temp_1[16] = temp_0[16];
      temp_1[17] = temp_0[17];
      temp_1[18] = temp_0[18];
      temp_1[19] = temp_0[19];
      temp_1[20] = temp_0[20] + temp_0[22];
      temp_1[21] = temp_0[21] + temp_0[23];
      temp_1[22] = temp_0[20] - temp_0[22];
      temp_1[23] = temp_0[21] - temp_0[23];
      temp_1[24] = temp_0[26] + temp_0[28];
      temp_1[25] = temp_0[27] + temp_0[29];
      temp_1[26] = temp_0[24] + temp_0[28];
      temp_1[27] = temp_0[25] + temp_0[29];
      temp_1[28] = temp_0[32] + temp_0[34];
      temp_1[29] = temp_0[33] + temp_0[35];
      temp_1[30] = temp_0[30] + temp_0[34];
      temp_1[31] = temp_0[31] + temp_0[35];

      /*  Stage s12  */

      temp_0[0] = temp_1[0];
      temp_0[1] = temp_1[1];
      temp_0[2] = temp_1[2];
      temp_0[3] = temp_1[3];
      temp_0[4] = temp_1[4];
      temp_0[5] = temp_1[6];
      temp_0[6] = temp_1[5];
      temp_0[7] = temp_1[7];
      temp_0[8] = temp_1[8];
      temp_0[9] = temp_1[10];
      temp_0[10] = temp_1[9];
      temp_0[11] = temp_1[11];
      temp_0[12] = temp_1[12];
      temp_0[13] = temp_1[13];
      temp_0[14] = temp_1[15];
      temp_0[15] = temp_1[14];
      temp_0[16] = temp_1[16];
      temp_0[17] = temp_1[18];
      temp_0[18] = temp_1[17];
      temp_0[19] = temp_1[19];
      temp_0[20] = temp_1[20];
      temp_0[21] = temp_1[21];
      temp_0[22] = temp_1[23];
      temp_0[23] = temp_1[22];
      temp_0[24] = temp_1[26];
      temp_0[25] = temp_1[25];
      temp_0[26] = temp_1[24];
      temp_0[27] = temp_1[27];
      temp_0[28] = temp_1[30];
      temp_0[29] = temp_1[29];
      temp_0[30] = temp_1[28];
      temp_0[31] = temp_1[31];

      /*  Stage s13  */

      temp_1[0] = temp_0[0] + temp_0[4];
      temp_1[1] = temp_0[1] + temp_0[5];
      temp_1[2] = temp_0[2] + temp_0[6];
      temp_1[3] = temp_0[3] + temp_0[7];
      temp_1[4] = temp_0[0] - temp_0[4];
      temp_1[5] = temp_0[1] - temp_0[5];
      temp_1[6] = temp_0[2] - temp_0[6];
      temp_1[7] = temp_0[3] - temp_0[7];
      temp_1[8] = temp_0[8];
      temp_1[9] = temp_0[9];
      temp_1[10] = temp_0[10];
      temp_1[11] = temp_0[11];
      temp_1[12] = temp_0[12];
      temp_1[13] = temp_0[13];
      temp_1[14] = temp_0[14];
      temp_1[15] = temp_0[15];
      temp_1[16] = temp_0[16];
      temp_1[17] = temp_0[17];
      temp_1[18] = temp_0[18];
      temp_1[19] = temp_0[19];
      temp_1[20] = temp_0[20];
      temp_1[21] = temp_0[21];
      temp_1[22] = temp_0[22];
      temp_1[23] = temp_0[23];
      temp_1[24] = temp_0[24] + temp_0[28];
      temp_1[25] = temp_0[25] + temp_0[29];
      temp_1[26] = temp_0[26] + temp_0[30];
      temp_1[27] = temp_0[27] + temp_0[31];
      temp_1[28] = temp_0[24] - temp_0[28];
      temp_1[29] = temp_0[25] - temp_0[29];
      temp_1[30] = temp_0[26] - temp_0[30];
      temp_1[31] = temp_0[27] - temp_0[31];

      /*  Stage s14  */

      temp_0[0] = temp_1[0];
      temp_0[1] = temp_1[1];
      temp_0[2] = temp_1[2];
      temp_0[3] = temp_1[3];
      temp_0[4] = temp_1[4];
      temp_0[5] = temp_1[5];
      temp_0[6] = temp_1[6];
      temp_0[7] = temp_1[7];
      temp_0[8] = temp_1[8];
      temp_0[9] = temp_1[12];
      temp_0[10] = temp_1[9];
      temp_0[11] = temp_1[13];
      temp_0[12] = temp_1[10];
      temp_0[13] = temp_1[14];
      temp_0[14] = temp_1[11];
      temp_0[15] = temp_1[15];
      temp_0[16] = temp_1[16];
      temp_0[17] = temp_1[20];
      temp_0[18] = temp_1[17];
      temp_0[19] = temp_1[21];
      temp_0[20] = temp_1[18];
      temp_0[21] = temp_1[22];
      temp_0[22] = temp_1[19];
      temp_0[23] = temp_1[23];
      temp_0[24] = temp_1[24];
      temp_0[25] = temp_1[25];
      temp_0[26] = temp_1[26];
      temp_0[27] = temp_1[27];
      temp_0[28] = temp_1[31];
      temp_0[29] = temp_1[30];
      temp_0[30] = temp_1[29];
      temp_0[31] = temp_1[28];

      /*  Stage s15  */

      temp_1[0] = temp_0[0] + temp_0[8];
      temp_1[1] = temp_0[1] + temp_0[9];
      temp_1[2] = temp_0[10] + temp_0[2];
      temp_1[3] = temp_0[11] + temp_0[3];
      temp_1[4] = temp_0[12] + temp_0[4];
      temp_1[5] = temp_0[13] + temp_0[5];
      temp_1[6] = temp_0[14] + temp_0[6];
      temp_1[7] = temp_0[15] + temp_0[7];
      temp_1[8] = temp_0[0] - temp_0[8];
      temp_1[9] = temp_0[1] - temp_0[9];
      temp_1[10] = temp_0[2] - temp_0[10];
      temp_1[11] = temp_0[3] - temp_0[11];
      temp_1[12] = temp_0[4] - temp_0[12];
      temp_1[13] = temp_0[5] - temp_0[13];
      temp_1[14] = temp_0[6] - temp_0[14];
      temp_1[15] = temp_0[7] - temp_0[15];
      temp_1[16] = temp_0[16];
      temp_1[17] = temp_0[24];
      temp_1[18] = temp_0[17];
      temp_1[19] = temp_0[25];
      temp_1[20] = temp_0[18];
      temp_1[21] = temp_0[26];
      temp_1[22] = temp_0[19];
      temp_1[23] = temp_0[27];
      temp_1[24] = temp_0[20];
      temp_1[25] = temp_0[28];
      temp_1[26] = temp_0[21];
      temp_1[27] = temp_0[29];
      temp_1[28] = temp_0[22];
      temp_1[29] = temp_0[30];
      temp_1[30] = temp_0[23];
      temp_1[31] = temp_0[31];

      /*  Stage s16  */

      current_elements[index_in_mem[0]] = temp_1[0] + temp_1[16];
      current_elements[index_in_mem[1]] = temp_1[1] + temp_1[17];
      current_elements[index_in_mem[2]] = temp_1[18] + temp_1[2];
      current_elements[index_in_mem[3]] = temp_1[19] + temp_1[3];
      current_elements[index_in_mem[4]] = temp_1[20] + temp_1[4];
      current_elements[index_in_mem[5]] = temp_1[21] + temp_1[5];
      current_elements[index_in_mem[6]] = temp_1[22] + temp_1[6];
      current_elements[index_in_mem[7]] = temp_1[23] + temp_1[7];
      current_elements[index_in_mem[8]] = temp_1[24] + temp_1[8];
      current_elements[index_in_mem[9]] = temp_1[25] + temp_1[9];
      current_elements[index_in_mem[10]] = temp_1[10] + temp_1[26];
      current_elements[index_in_mem[11]] = temp_1[11] + temp_1[27];
      current_elements[index_in_mem[12]] = temp_1[12] + temp_1[28];
      current_elements[index_in_mem[13]] = temp_1[13] + temp_1[29];
      current_elements[index_in_mem[14]] = temp_1[14] + temp_1[30];
      current_elements[index_in_mem[15]] = temp_1[15] + temp_1[31];
      current_elements[index_in_mem[16]] = temp_1[0] - temp_1[16];
      current_elements[index_in_mem[17]] = temp_1[1] - temp_1[17];
      current_elements[index_in_mem[18]] = temp_1[2] - temp_1[18];
      current_elements[index_in_mem[19]] = temp_1[3] - temp_1[19];
      current_elements[index_in_mem[20]] = temp_1[4] - temp_1[20];
      current_elements[index_in_mem[21]] = temp_1[5] - temp_1[21];
      current_elements[index_in_mem[22]] = temp_1[6] - temp_1[22];
      current_elements[index_in_mem[23]] = temp_1[7] - temp_1[23];
      current_elements[index_in_mem[24]] = temp_1[8] - temp_1[24];
      current_elements[index_in_mem[25]] = temp_1[9] - temp_1[25];
      current_elements[index_in_mem[26]] = temp_1[10] - temp_1[26];
      current_elements[index_in_mem[27]] = temp_1[11] - temp_1[27];
      current_elements[index_in_mem[28]] = temp_1[12] - temp_1[28];
      current_elements[index_in_mem[29]] = temp_1[13] - temp_1[29];
      current_elements[index_in_mem[30]] = temp_1[14] - temp_1[30];
      current_elements[index_in_mem[31]] = temp_1[15] - temp_1[31];
    }
  }

  template <typename S, typename E>
  void NttCpu<S, E>::hierarchy_0_dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates) // R --> N
  {
    const uint32_t subntt_size_log =
      this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                    [ntt_task_cordinates.hierarchy_0_layer_idx];
    const uint64_t subntt_size = 1 << subntt_size_log;
    // std::cout << "radix2_dit_ntt, subntt_size: " << subntt_size << std::endl;

    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;

    std::vector<uint32_t> index_in_mem(subntt_size);
    for (uint32_t i = 0; i < subntt_size; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_cordinates, i);
    }
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);
      for (uint32_t len = 2; len <= subntt_size; len <<= 1) {
        uint32_t half_len = len / 2;
        uint32_t step = (subntt_size / len) * (this->domain_max_size >> subntt_size_log);
        for (uint32_t i = 0; i < subntt_size; i += len) {
          for (uint32_t j = 0; j < half_len; ++j) {
            uint32_t tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
            uint64_t u_mem_idx = index_in_mem[i + j];
            uint64_t v_mem_idx = index_in_mem[i + j + half_len];
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
    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements =
        this->config.columns_batch ? elements + batch : elements + batch * (this->ntt_sub_logn.size);
      for (uint32_t len = subntt_size; len >= 2; len >>= 1) {
        uint32_t half_len = len / 2;
        uint32_t step = (subntt_size / len) * (this->domain_max_size / subntt_size);
        for (uint32_t i = 0; i < subntt_size; i += len) {
          for (uint32_t j = 0; j < half_len; ++j) {
            uint32_t tw_idx = (this->direction == NTTDir::kForward) ? j * step : this->domain_max_size - j * step;
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
  void NttCpu<S, E>::coset_mul(E* elements, uint32_t coset_stride, const std::unique_ptr<S[]>& arbitrary_coset)
  {
    uint64_t size = this->ntt_sub_logn.size;
    uint32_t batch_stride = this->config.columns_batch ? this->config.batch_size : 1;
    const bool needs_reorder_input = this->direction == NTTDir::kForward && (this->ntt_sub_logn.logn > HIERARCHY_1);

    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * size;

      for (uint64_t i = 1; i < size; ++i) {
        uint64_t idx = i;

        // Adjust the index if reorder logic was applied on the input
        if (needs_reorder_input) {
          uint32_t cur_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
          uint32_t next_ntt_log_size = this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
          uint32_t subntt_idx = i >> cur_ntt_log_size;
          uint32_t element = i & ((1 << cur_ntt_log_size) - 1);
          idx = subntt_idx + (element << next_ntt_log_size);
        }

        // Apply coset multiplication based on the available coset information
        if (arbitrary_coset) {
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * arbitrary_coset[idx];
        } else {
          uint32_t twiddle_idx = coset_stride * idx;
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
   * @return uint32_t Returns the coset stride if found in the precomputed twiddles. If an arbitrary coset is calculated,
   *             returns 0 (since the stride is not applicable in this case).
   */

  template <typename S, typename E>
  uint32_t NttCpu<S, E>::find_or_generate_coset(std::unique_ptr<S[]>& arbitrary_coset)
  {
    uint32_t coset_stride = 0;

    if (config.coset_gen != S::one()) {
      try {
        coset_stride =
          CpuNttDomain<S>::s_ntt_domain.get_coset_stride(config.coset_gen); // Coset generator found in twiddles
      } catch (const std::out_of_range& oor) { // Coset generator not found in twiddles. Calculating arbitrary coset
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen =
          direction == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
        for (uint32_t i = 1; i <= domain_max_size; i++) {
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
    const uint32_t sntt_size = 1 << this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
    const uint32_t nof_sntts = 1 << this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
    const uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    const uint64_t temp_elements_size = this->ntt_sub_logn.size * this->config.batch_size;

    auto temp_elements = std::make_unique<E[]>(temp_elements_size);
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* cur_layer_output = this->config.columns_batch ? elements + batch : elements + batch * this->ntt_sub_logn.size;
      E* cur_temp_elements = this->config.columns_batch ? temp_elements.get() + batch
                                                        : temp_elements.get() + batch * this->ntt_sub_logn.size;
      for (uint32_t sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (uint32_t elem = 0; elem < sntt_size; elem++) {
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
    uint32_t subntt_idx;
    uint32_t element;
    uint32_t s0 = is_top_hirarchy
               ? this->ntt_sub_logn.hierarchy_1_layers_sub_logn[0]
               : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0];
    uint32_t s1 = is_top_hirarchy
               ? this->ntt_sub_logn.hierarchy_1_layers_sub_logn[1]
               : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
    uint32_t s2 = is_top_hirarchy
               ? this->ntt_sub_logn.hierarchy_1_layers_sub_logn[2]
               : this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
    uint32_t p0, p1, p2;
    const uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    uint32_t rep = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t tw_idx = 0;
    E* hierarchy_1_subntt_output =
      elements +
      stride * (ntt_task_cordinates.hierarchy_1_subntt_idx
                << this->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                               // subntt_size
    for (uint32_t batch = 0; batch < rep; ++batch) {
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
    uint32_t hierarchy_0_subntt_size =
      1 << NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                                 [ntt_task_cordinates.hierarchy_0_layer_idx];
    uint32_t hierarchy_0_nof_subntts =
      1 << NttCpu<S, E>::ntt_sub_logn
             .hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0]; // only relevant for layer 1
    uint32_t i, j, i_0;
    uint32_t ntt_size =
      ntt_task_cordinates.hierarchy_0_layer_idx == 0
        ? 1
            << (NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0] +
                NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1])
        : 1
            << (NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0] +
                NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1] +
                NttCpu<S, E>::ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2]);
    uint32_t stride = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t original_size = (1 << NttCpu<S, E>::ntt_sub_logn.logn);
    for (uint32_t batch = 0; batch < this->config.batch_size; ++batch) {
      E* hierarchy_1_subntt_elements =
        elements +
        stride * (ntt_task_cordinates.hierarchy_1_subntt_idx
                  << NttCpu<S, E>::ntt_sub_logn
                       .hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
      E* elements_of_current_batch = this->config.columns_batch ? hierarchy_1_subntt_elements + batch
                                                                : hierarchy_1_subntt_elements + batch * original_size;
      for (uint32_t elem = 0; elem < hierarchy_0_subntt_size; elem++) {
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
    uint32_t nof_hierarchy_0_layers =
      (this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2] != 0)   ? 3
      : (this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1] != 0) ? 2
                                                                                                            : 1;
    uint32_t log_nof_blocks;
    uint32_t log_nof_subntts;
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

    uint32_t offset = this->config.columns_batch ? this->config.batch_size : 1;
    E* current_input =
      input +
      offset * (ntt_task_cordinates.hierarchy_1_subntt_idx
                << NttCpu<S, E>::ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                               // subntt_size

    const uint32_t subntt_size_log =
      this->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx]
                                                    [ntt_task_cordinates.hierarchy_0_layer_idx];
    switch (subntt_size_log)
    {
    case 3:
      ntt8win(current_input, ntt_task_cordinates);
      break;
    case 4:
      ntt16win(current_input, ntt_task_cordinates);
      break;
    case 5:
      ntt32win(current_input, ntt_task_cordinates);
      break;
    default:
      this->reorder_by_bit_reverse(ntt_task_cordinates, current_input, false);
      this->hierarchy_0_dit_ntt(current_input, ntt_task_cordinates); // R --> N
      break;
    }

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
    TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, uint32_t hierarchy_1_layer_idx)
  {
    NttTask<S, E>* task_slot = nullptr;
    NttTaskParams<S, E> params;

    uint32_t nof_subntts_l1 = 1
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
  template <> // MIKI - am I waisting time here?
  struct hash<ntt_cpu::NttTaskCordinates> {
    std::size_t operator()(const ntt_cpu::NttTaskCordinates& key) const
    {
      // Combine hash values of the members using a simple hash combiner
      return ((std::hash<uint32_t>()(key.hierarchy_1_layer_idx) ^ (std::hash<uint32_t>()(key.hierarchy_1_subntt_idx) << 1)) >>
              1) ^
             (std::hash<uint32_t>()(key.hierarchy_0_layer_idx) << 1) ^ (std::hash<uint32_t>()(key.hierarchy_0_block_idx) << 1) ^
             (std::hash<uint32_t>()(key.hierarchy_0_subntt_idx) << 1);
    }
  };
} // namespace std
