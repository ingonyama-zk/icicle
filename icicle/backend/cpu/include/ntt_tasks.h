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

#define H1 15

using namespace field_config;
using namespace icicle;
namespace ntt_cpu {

  /**
   * @brief Defines the log sizes of sub-NTTs for different problem sizes.
   *
   * `layers_sub_logn` specifies the log sizes for up to three layers (hierarcy1 or hierarcy0) in the NTT computation.
   * - The outer index represents the log size (`logn`) of the original NTT problem.
   * - Each inner array contains three integers corresponding to the log sizes for each hierarchical layer.
   *
   * Example: `layers_sub_logn[14] = {14, 13, 0}` means for `logn = 14`, the sub-NTT log sizes are 14 for the first
   * layer, 13 for the second, and 0 for the third.
   */
  constexpr uint32_t layers_sub_logn[31][3] = {
    // {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {1, 1, 1},   {2, 2, 0},   {2, 2, 1},   {3, 2, 1},   {4, 3, 0},
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

  /**
   * @brief Represents the coordinates of a task in the NTT hierarchy.
   * This struct holds indices that identify the position of a task within the NTT computation hierarchy.
   *
   * @param h1_layer_idx Index of the h1 layer.
   * @param h1_subntt_idx Index of the sub-NTT within the h1 layer.
   * @param h0_layer_idx Index of the h0 layer.
   * @param h0_block_idx Index of the block within the h0 layer.
   * @param h0_subntt_idx Index of the sub-NTT within the h0 block.
   *
   * @method bool operator==(const NttTaskCordinates& other) const Compares two task coordinates for equality.
   */
  struct NttTaskCordinates {
    int h1_layer_idx = 0;
    int h1_subntt_idx = 0;
    int h0_layer_idx = 0;
    int h0_block_idx = 0;
    int h0_subntt_idx = 0;

    bool operator==(const NttTaskCordinates& other) const
    {
      return h1_layer_idx == other.h1_layer_idx && h1_subntt_idx == other.h1_subntt_idx &&
             h0_layer_idx == other.h0_layer_idx && h0_block_idx == other.h0_block_idx &&
             h0_subntt_idx == other.h0_subntt_idx;
    }
  };

  /**
   * @brief Represents the log sizes of sub-NTTs in the NTT computation hierarchy.
   *
   * This struct stores the log sizes of the sub-NTTs for both h0 and h1 hierarchy layers,
   * based on the overall log size (`logn`) of the NTT problem.
   *
   * @param logn The log size of the entire NTT problem.
   * @param size The size of the NTT problem, calculated as `1 << logn`.
   * @param h0_layers_sub_logn Log sizes of sub-NTTs for h0 layers.
   * @param h1_layers_sub_logn Log sizes of sub-NTTs for h1 layers.
   *
   * @method NttSubLogn(int logn) Initializes the struct based on the given `logn`.
   */
  struct NttSubLogn {
    int logn;                                         // Original log_size of the problem
    uint64_t size;                                    // Original log_size of the problem
    std::vector<std::vector<int>> h0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<int> h1_layers_sub_logn;              // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubLogn(int logn) : logn(logn)
    {
      size = 1 << logn;
      if (logn > H1) {
        // Initialize h1_layers_sub_logn
        h1_layers_sub_logn = std::vector<int>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn]));
        // Initialize h0_layers_sub_logn
        h0_layers_sub_logn = {
          std::vector<int>(
            std::begin(layers_sub_logn[h1_layers_sub_logn[0]]), std::end(layers_sub_logn[h1_layers_sub_logn[0]])),
          std::vector<int>(
            std::begin(layers_sub_logn[h1_layers_sub_logn[1]]), std::end(layers_sub_logn[h1_layers_sub_logn[1]]))};
      } else {
        h1_layers_sub_logn = {0, 0, 0};
        h0_layers_sub_logn = {
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
    // double duration_total=0;

    NttCpu(int logn, NTTDir direction, const NTTConfig<S>& config, int domain_max_size, const S* twiddles)
        : ntt_sub_logn(logn), direction(direction), config(config), domain_max_size(domain_max_size), twiddles(twiddles)
    {
    }

    eIcicleError reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy);
    eIcicleError copy_and_reorder_if_needed(const E* input, E* output);
    void h0_dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles);
    void dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles);
    eIcicleError
    coset_mul(E* elements, const S* twiddles, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset);
    int find_or_generate_coset(std::unique_ptr<S[]>& arbitrary_coset);
    void h1_reorder(E* elements);
    eIcicleError
    reorder_and_refactor_if_needed(E* elements, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy);
    eIcicleError
    hierarchy1_push_tasks(E* input, NttTaskCordinates ntt_task_cordinates, NttTasksManager<S, E>& ntt_tasks_manager);
    eIcicleError h0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates);
    eIcicleError handle_pushed_tasks(
      TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, int h1_layer_idx);

  private:
    uint64_t bit_reverse(uint64_t n, int logn);
    uint64_t idx_in_mem(NttTaskCordinates ntt_task_cordinates, int element);
    void refactor_output_h0(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles);
  }; // class NttCpu

  /**
   * @brief Manages task dependency counters for NTT computation, tracking readiness of tasks to execute.
   *
   * This class tracks and manages counters for tasks within the NTT hierarchy, determining when tasks are ready to
   * execute based on the completion of their dependencies.
   *
   * @param h1_layer_idx Index of the h1 layer this counter set belongs to.
   * @param nof_h0_layers Number of h0 layers in the current h1 layer.
   * @param nof_pointing_to_counter Number of counters pointing to each h0 layer.
   * @param h0_counters A 3D vector of shared pointers to counters for each sub-NTT in h0 layers.
   * @param h1_counters A vector of shared pointers to counters for each sub-NTT in h1 layers, used to signal when an
   * h1_subntt is ready for reordering.
   *
   * @method TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx) Constructor that initializes the
   * counters based on NTT structure.
   * @method bool decrement_counter(NttTaskCordinates ntt_task_cordinates) Decrements the counter for a given task and
   * returns true if the task is ready to execute.
   * @method int get_nof_pointing_to_counter(int h0_layer_idx) Returns the number of counters pointing to the given h0
   * layer.
   * @method int get_nof_h0_layers() Returns the number of h0 layers in the current h1 layer.
   */
  class TasksDependenciesCounters
  {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx);

    // Function to decrement the counter for a given task and check if it is ready to execute. if so, return true
    bool decrement_counter(NttTaskCordinates ntt_task_cordinates);
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
        // if all h0_subntts are done, and at least 2 layers in hierarchy 0 - reorder the subntt's output
        if (ntt_cpu->config.columns_batch) {
          ntt_cpu->reorder_and_refactor_if_needed(input, ntt_task_cordinates, false);
        } else {
          for (int b = 0; b < ntt_cpu->config.batch_size; b++) {
            ntt_cpu->reorder_and_refactor_if_needed(
              input + b * (1 << (ntt_cpu->ntt_sub_logn.logn)), ntt_task_cordinates, false);
          }
        }
      } else {
        ntt_cpu->h0_cpu_ntt(input, ntt_task_cordinates);
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

  template <typename S = scalar_t, typename E = scalar_t>
  class NttTasksManager
  {
  public:
    NttTasksManager(int logn);

    // Add a new task to the ntt_task_manager
    eIcicleError push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder);

    // Set a task as completed and update dependencies
    eIcicleError set_task_as_completed(NttTask<S, E>& completed_task, int nof_subntts_l2);

    bool tasks_to_do() { return !available_tasks_list.empty() || !waiting_tasks_map.empty(); }

    bool available_tasks() { return !available_tasks_list.empty(); }

    NttTaskParams<S, E> get_available_task() { return available_tasks_list.front(); }

    eIcicleError erase_task_from_available_tasks_list()
    {
      available_tasks_list.pop_front();
      return eIcicleError::SUCCESS;
    }

  private:
    std::vector<TasksDependenciesCounters> counters; // Dependencies counters by layer
    std::deque<NttTaskParams<S, E>> available_tasks_list;
    std::unordered_map<NttTaskCordinates, NttTaskParams<S, E>> waiting_tasks_map;
  };

  //////////////////////////// TasksDependenciesCounters Implementation ////////////////////////////

  TasksDependenciesCounters::TasksDependenciesCounters(NttSubLogn ntt_sub_logn, int h1_layer_idx)
      : h0_counters(
          1
          << ntt_sub_logn.h1_layers_sub_logn[1 - h1_layer_idx]), // nof_h1_subntts = h1_layers_sub_logn[1-h1_layer_idx].
        h1_counters(1 << ntt_sub_logn.h1_layers_sub_logn[1 - h1_layer_idx])
  { // Initialize h1_counters with N0 * N1

    nof_h0_layers =
      ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][2] ? 3 : (ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1] ? 2 : 1);
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
      h0_counters[h1_subntt_idx].resize(3); // 3 layers (0, 1, 2)
      // Initialize counters for layer 0 - 1 counter1 initialized with 0.
      h0_counters[h1_subntt_idx][0].resize(1);
      h0_counters[h1_subntt_idx][0][0] = std::make_shared<int>(0); //[h1_subntt_idx][h0_layer_idx][h0_counter_idx]

      if (nof_h0_layers > 1) {
        // Initialize counters for layer 1 - N2 counters initialized with N1.
        h0_counters[h1_subntt_idx][1].resize(l1_nof_counters);
        for (int counter_idx = 0; counter_idx < l1_nof_counters; ++counter_idx) {
          h0_counters[h1_subntt_idx][1][counter_idx] = std::make_shared<int>(l1_counter_size);
        }
      }
      if (nof_h0_layers > 2) {
        // Initialize counters for layer 2 - N0 counters initialized with N2.
        h0_counters[h1_subntt_idx][2].resize(l2_nof_counters);
        for (int counter_idx = 0; counter_idx < l2_nof_counters; ++counter_idx) {
          h0_counters[h1_subntt_idx][2][counter_idx] = std::make_shared<int>(l2_counter_size);
        }
      }
      // Initialize h1_counters with N0 * N1
      int h1_counter_size = nof_h0_layers == 3 ? (1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0]) *
                                                   (1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][1])
                            : nof_h0_layers == 2 ? (1 << ntt_sub_logn.h0_layers_sub_logn[h1_layer_idx][0])
                                                 : 0;
      h1_counters[h1_subntt_idx] = std::make_shared<int>(h1_counter_size);
    }
  }

  bool TasksDependenciesCounters::decrement_counter(NttTaskCordinates task_c)
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

  //////////////////////////// NttTasksManager Implementation ////////////////////////////

  template <typename S, typename E>
  NttTasksManager<S, E>::NttTasksManager(int logn)
      : counters(logn > H1 ? 2 : 1, TasksDependenciesCounters(NttSubLogn(logn), 0))
  {
    if (logn > H1) { counters[1] = TasksDependenciesCounters(NttSubLogn(logn), 1); }
  }

  template <typename S, typename E>
  eIcicleError NttTasksManager<S, E>::push_task(NttCpu<S, E>* ntt_cpu, E* input, NttTaskCordinates task_c, bool reorder)
  {
    // Create a new NttTaskParams and add it to the available_tasks_list
    NttTaskParams<S, E> params = {ntt_cpu, input, task_c, reorder};
    if (task_c.h0_layer_idx == 0) {
      available_tasks_list.push_back(params);
    } else {
      waiting_tasks_map[task_c] = params; // Add to map
    }
    return eIcicleError::SUCCESS;
  }

  // Function to set a task as completed and update dependencies
  template <typename S, typename E>
  eIcicleError NttTasksManager<S, E>::set_task_as_completed(NttTask<S, E>& completed_task, int nof_subntts_l2)
  {
    ntt_cpu::NttTaskCordinates task_c = completed_task.get_coordinates();
    int nof_h0_layers = counters[task_c.h1_layer_idx].get_nof_h0_layers();
    // Update dependencies in counters
    if (counters[task_c.h1_layer_idx].decrement_counter(task_c)) {
      if (task_c.h0_layer_idx < nof_h0_layers - 1) {
        int nof_pointing_to_counter =
          (task_c.h0_layer_idx == nof_h0_layers - 1)
            ? 1
            : counters[task_c.h1_layer_idx].get_nof_pointing_to_counter(task_c.h0_layer_idx + 1);
        int stride = nof_subntts_l2 / nof_pointing_to_counter;
        for (int i = 0; i < nof_pointing_to_counter; i++) {
          NttTaskCordinates next_task_c =
            task_c.h0_layer_idx == 0
              ? NttTaskCordinates{task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx + 1, task_c.h0_block_idx, i}
              /*task_c.h0_layer_idx==1*/
              : NttTaskCordinates{
                  task_c.h1_layer_idx, task_c.h1_subntt_idx, task_c.h0_layer_idx + 1,
                  (task_c.h0_subntt_idx + stride * i), 0};
          auto it = waiting_tasks_map.find(next_task_c);
          if (it != waiting_tasks_map.end()) {
            available_tasks_list.push_back(it->second);
            waiting_tasks_map.erase(it);
          } else {
            ICICLE_LOG_ERROR << "Task not found in waiting_tasks_map: h0_layer_idx: " << next_task_c.h0_layer_idx
                             << ", h0_block_idx: " << next_task_c.h0_block_idx
                             << ", h0_subntt_idx: " << next_task_c.h0_subntt_idx;
          }
        }
      } else {
        // Reorder the output
        NttTaskCordinates next_task_c = {task_c.h1_layer_idx, task_c.h1_subntt_idx, nof_h0_layers, 0, 0};
        auto it = waiting_tasks_map.find(next_task_c);
        if (it != waiting_tasks_map.end()) {
          available_tasks_list.push_back(it->second);
          waiting_tasks_map.erase(it);
        } else {
          ICICLE_LOG_ERROR << "Task not found in waiting_tasks_map";
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
   * The function supports different layer configurations (`h0_layer_idx`) within the sub-NTT,
   * and returns the appropriate memory index based on the element's position within the hierarchy.
   *
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT hierarchy.
   * @param element_idx The specific element index within the sub-NTT.
   * @return uint64_t The computed memory index for the given element.
   */

  template <typename S, typename E>
  uint64_t NttCpu<S, E>::idx_in_mem(NttTaskCordinates ntt_task_cordinates, int element_idx)
  {
    int s0 = this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
    int s1 = this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
    int s2 = this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
    switch (ntt_task_cordinates.h0_layer_idx) {
    case 0:
      return ntt_task_cordinates.h0_block_idx + ((ntt_task_cordinates.h0_subntt_idx + (element_idx << s1)) << s2);
    case 1:
      return ntt_task_cordinates.h0_block_idx + ((element_idx + (ntt_task_cordinates.h0_subntt_idx << s1)) << s2);
    case 2:
      return ((ntt_task_cordinates.h0_block_idx << (s1 + s2)) & ((1 << (s0 + s1 + s2)) - 1)) +
             (((ntt_task_cordinates.h0_block_idx << (s1 + s2)) >> (s0 + s1 + s2)) << s2) + element_idx;
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
   * @return eIcicleError Returns `SUCCESS` if the reordering is successful, otherwise returns an error code.
   */

  template <typename S, typename E>
  eIcicleError
  NttCpu<S, E>::reorder_by_bit_reverse(NttTaskCordinates ntt_task_cordinates, E* elements, bool is_top_hirarchy)
  {
    uint64_t subntt_size =
      is_top_hirarchy ? (this->ntt_sub_logn.size)
                      : 1 << this->ntt_sub_logn
                               .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int subntt_log_size =
      is_top_hirarchy
        ? (this->ntt_sub_logn.logn)
        : this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
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
            return eIcicleError::INVALID_ARGUMENT;
          }
        }
      }
    }
    return eIcicleError::SUCCESS;
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
   * @note - If `logn > H1`, there is an additional level of hierarchy due to sub-NTTs, requiring extra reordering
   *       - If `logn <= H1`, only bit-reversal reordering is applied if configured.
   *       - If no reordering is needed, the input data is directly copied to the output.
   */

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::copy_and_reorder_if_needed(const E* input, E* output)
  {
    const uint64_t total_memory_size = this->ntt_sub_logn.size * config.batch_size;
    const int stride = config.columns_batch ? config.batch_size : 1;
    const int logn = static_cast<int>(std::log2(this->ntt_sub_logn.size));
    const bool bit_rev = config.ordering == Ordering::kRN || config.ordering == Ordering::kRR;

    if (logn > H1) {
      // Apply input's reorder logic depending on the configuration
      int cur_ntt_log_size = this->ntt_sub_logn.h1_layers_sub_logn[0];
      int next_ntt_log_size = this->ntt_sub_logn.h1_layers_sub_logn[1];

      for (int batch = 0; batch < config.batch_size; ++batch) {
        const E* input_batch = config.columns_batch ? (input + batch) : (input + batch * this->ntt_sub_logn.size);
        E* output_batch = config.columns_batch ? (output + batch) : (output + batch * this->ntt_sub_logn.size);

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
        E* output_batch = config.columns_batch ? (output + batch) : (output + batch * this->ntt_sub_logn.size);

        for (uint64_t i = 0; i < this->ntt_sub_logn.size; ++i) {
          uint64_t rev = bit_reverse(i, logn);
          output_batch[stride * i] = input_batch[stride * rev];
        }
      }
    } else {
      // Just copy, no reordering needed
      std::copy(input, input + total_memory_size, output);
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
   * @param twiddles The array of precomputed twiddle factors used in the NTT transformation.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::h0_dit_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles) // R --> N
  {
    const int subntt_size_log =
      this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
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
            E v = current_elements[v_mem_idx] * twiddles[tw_idx];
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
   * @param twiddles The array of precomputed twiddle factors used in the NTT transformation.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::dif_ntt(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles)
  {
    uint64_t subntt_size =
      1 << this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
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
            current_elements[v_mem_idx] = (u - v) * twiddles[tw_idx];
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
   * @param twiddles A pointer to the twiddle factors used for multiplication if `arbitrary_coset` is not provided.
   * @param coset_stride The stride used to select the appropriate twiddle factor. This is computed based on the coset
   * generator.
   * @param arbitrary_coset A unique pointer to an array of arbitrary coset values generated if the coset generator is
   * not found in the twiddles.
   *
   * @return eIcicleError Returns `eIcicleError::SUCCESS` if the coset multiplication is applied successfully. Returns
   * an error code otherwise.
   *
   * @note This function assumes that the input data may have undergone reordering, and it adjusts the indices used for
   *       coset multiplication accordingly. The function handles both the cases where `twiddles` are used and where
   *       an `arbitrary_coset` is used.
   */
  template <typename S, typename E>
  eIcicleError
  NttCpu<S, E>::coset_mul(E* elements, const S* twiddles, int coset_stride, const std::unique_ptr<S[]>& arbitrary_coset)
  {
    uint64_t size = this->ntt_sub_logn.size;
    int batch_stride = this->config.columns_batch ? this->config.batch_size : 1;
    const int logn = static_cast<int>(std::log2(size));
    const bool needs_reorder_input = this->direction == NTTDir::kForward && (logn > H1);

    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* current_elements = this->config.columns_batch ? elements + batch : elements + batch * size;

      for (uint64_t i = 1; i < size; ++i) {
        uint64_t idx = i;

        // Adjust the index if reorder logic was applied on the input
        if (needs_reorder_input) {
          int cur_ntt_log_size = this->ntt_sub_logn.h1_layers_sub_logn[0];
          int next_ntt_log_size = this->ntt_sub_logn.h1_layers_sub_logn[1];
          int subntt_idx = i >> cur_ntt_log_size;
          int element = i & ((1 << cur_ntt_log_size) - 1);
          idx = subntt_idx + (element << next_ntt_log_size);
        }

        // Apply coset multiplication based on the available coset information
        if (arbitrary_coset) {
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * arbitrary_coset[idx];
        } else if (coset_stride != 0) {
          int twiddle_idx = coset_stride * idx;
          twiddle_idx = this->direction == NTTDir::kForward ? twiddle_idx : this->domain_max_size - twiddle_idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * twiddles[twiddle_idx];
        }
      }
    }
    return eIcicleError::SUCCESS;
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
          CpuNttDomain<S>::s_ntt_domain.coset_index.at(config.coset_gen); // Coset generator found in twiddles
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
   * @param twiddles A pointer to the twiddle factors array.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::h1_reorder(E* elements)
  {
    const int sntt_size = 1 << this->ntt_sub_logn.h1_layers_sub_logn[1];
    const int nof_sntts = 1 << this->ntt_sub_logn.h1_layers_sub_logn[0];
    const int stride = this->config.columns_batch ? this->config.batch_size : 1;
    const uint64_t temp_elements_size = this->ntt_sub_logn.size * this->config.batch_size;

    auto temp_elements = std::make_unique<E[]>(temp_elements_size);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* cur_layer_output = this->config.columns_batch ? elements + batch : elements + batch * this->ntt_sub_logn.size;
      E* cur_temp_elements = this->config.columns_batch ? temp_elements.get() + batch
                                                        : temp_elements.get() + batch * this->ntt_sub_logn.size;
      for (int sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (int elem = 0; elem < sntt_size; elem++) {
          // uint64_t tw_idx = (this->direction == NTTDir::kForward)
          //                     ? ((this->domain_max_size >> this->ntt_sub_logn.logn) * sntt_idx * elem)
          //                     : this->domain_max_size - ((this->domain_max_size >> this->ntt_sub_logn.logn) *
          //                     sntt_idx * elem);
          // std::cout << "h1_subntt_idx=\t" << elem << std::endl;
          // std::cout << "cur_layer_output[" << stride * (elem * nof_sntts + sntt_idx) << "] = " <<
          // cur_layer_output[stride * (elem * nof_sntts + sntt_idx)] << std::endl; std::cout << "twiddles[" << tw_idx
          // << "] = " << twiddles[tw_idx] << std::endl; cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
          // cur_layer_output[stride * (elem * nof_sntts + sntt_idx)] * twiddles[tw_idx];
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
   * moving on to the next h1 layer.
   *
   * The reordering process involves reshuffling elements within the output array to match the required
   * structure, taking into account the sub-NTT sizes and indices.
   *
   * @param elements The array where the reordered and potentially refactored data will be stored.
   * @param ntt_task_cordinates The coordinates specifying the current task within the NTT computation hierarchy.
   * @param is_top_hirarchy A boolean indicating whether the function is operating at the top-level hierarchy (between
   * layers of h1).
   *
   */
  template <typename S, typename E>
  eIcicleError
  NttCpu<S, E>::reorder_and_refactor_if_needed(E* elements, NttTaskCordinates ntt_task_cordinates, bool is_top_hirarchy)
  {
    bool is_only_h0 = this->ntt_sub_logn.h1_layers_sub_logn[0] == 0;
    const bool refactor_pre_h1_next_layer =
      (!is_only_h0) && (!is_top_hirarchy) && (ntt_task_cordinates.h1_layer_idx == 0);
    // const bool refactor_pre_h1_next_layer = false;
    uint64_t size = (is_top_hirarchy || is_only_h0)
                      ? this->ntt_sub_logn.size
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
    const int stride = this->config.columns_batch ? this->config.batch_size : 1;
    int rep = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t tw_idx = 0;
    E* h1_subntt_output =
      elements +
      stride * (ntt_task_cordinates.h1_subntt_idx
                << NttCpu<S, E>::ntt_sub_logn
                     .h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size //TODO
                                                                             // - NttCpu or this?
    for (int batch = 0; batch < rep; ++batch) {
      E* current_elements = this->config.columns_batch
                              ? h1_subntt_output + batch
                              : h1_subntt_output; // if columns_batch=false, then output is already shifted by
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
        if (refactor_pre_h1_next_layer) {
          tw_idx =
            (this->direction == NTTDir::kForward)
              ? ((this->domain_max_size >> this->ntt_sub_logn.logn) * ntt_task_cordinates.h1_subntt_idx * new_idx)
              : this->domain_max_size -
                  ((this->domain_max_size >> this->ntt_sub_logn.logn) * ntt_task_cordinates.h1_subntt_idx * new_idx);
          current_temp_output[stride * new_idx] = current_elements[stride * i] * this->twiddles[tw_idx];
        } else {
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
    }
    std::copy(temp_output.get(), temp_output.get() + temp_output_size, h1_subntt_output);
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Refactors the output of an h0 sub-NTT after the NTT operation.
   *
   * This function refactors the output of an h0 sub-NTT by applying twiddle factors to the elements
   * based on their indices. It prepares the data for further processing in subsequent layers of the NTT hierarchy.
   * Accesses corrected memory addresses, because reordering between layers of hierarchy 0 was skipped.
   *
   * @param elements The array of elements that have been transformed by the NTT.
   * @param ntt_task_cordinates The coordinates specifying the sub-NTT within the NTT hierarchy.
   * @param twiddles The array of precomputed twiddle factors used for the refactoring.
   */

  template <typename S, typename E>
  void NttCpu<S, E>::refactor_output_h0(E* elements, NttTaskCordinates ntt_task_cordinates, const S* twiddles)
  {
    int h0_subntt_size = 1 << NttCpu<S, E>::ntt_sub_logn
                                .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx];
    int h0_nof_subntts = 1 << NttCpu<S, E>::ntt_sub_logn
                                .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]; // only relevant for layer 1
    int i, j, i_0;
    int ntt_size = ntt_task_cordinates.h0_layer_idx == 0
                     ? 1
                         << (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
                             NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1])
                     : 1
                         << (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
                             NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1] +
                             NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]);
    int stride = this->config.columns_batch ? this->config.batch_size : 1;
    uint64_t original_size = (1 << NttCpu<S, E>::ntt_sub_logn.logn);
    for (int batch = 0; batch < this->config.batch_size; ++batch) {
      E* h1_subntt_elements =
        elements +
        stride * (ntt_task_cordinates.h1_subntt_idx
                  << NttCpu<S, E>::ntt_sub_logn
                       .h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size
      E* elements_of_current_batch =
        this->config.columns_batch ? h1_subntt_elements + batch : h1_subntt_elements + batch * original_size;
      for (int elem = 0; elem < h0_subntt_size; elem++) {
        uint64_t elem_mem_idx = stride * idx_in_mem(ntt_task_cordinates, elem);
        i = (ntt_task_cordinates.h0_layer_idx == 0) ? elem : elem * h0_nof_subntts + ntt_task_cordinates.h0_subntt_idx;
        j = (ntt_task_cordinates.h0_layer_idx == 0) ? ntt_task_cordinates.h0_subntt_idx
                                                    : ntt_task_cordinates.h0_block_idx;
        uint64_t tw_idx = (this->direction == NTTDir::kForward)
                            ? ((this->domain_max_size / ntt_size) * j * i)
                            : this->domain_max_size - ((this->domain_max_size / ntt_size) * j * i);
        elements_of_current_batch[elem_mem_idx] = elements_of_current_batch[elem_mem_idx] * twiddles[tw_idx];
      }
    }
  }

  /**
   * @brief Pushes tasks for the h1 hierarchy of the NTT computation.
   *
   * This function organizes and pushes tasks for the h1 hierarchy level of the NTT computation
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
    int nof_h0_layers = (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2] != 0)   ? 3
                        : (NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1] != 0) ? 2
                                                                                                                    : 1;
    for (ntt_task_cordinates.h0_layer_idx = 0;
         ntt_task_cordinates.h0_layer_idx <
         NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx].size();
         ntt_task_cordinates.h0_layer_idx++) {
      if (ntt_task_cordinates.h0_layer_idx == 0) {
        int log_nof_subntts = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
        int log_nof_blocks = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks);
             ntt_task_cordinates.h0_block_idx++) {
          for (ntt_task_cordinates.h0_subntt_idx = 0; ntt_task_cordinates.h0_subntt_idx < (1 << log_nof_subntts);
               ntt_task_cordinates.h0_subntt_idx++) {
            ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
          }
        }
      }
      if (
        ntt_task_cordinates.h0_layer_idx == 1 &&
        NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]) {
        int log_nof_subntts = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0];
        int log_nof_blocks = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2];
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
        NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]) {
        ntt_task_cordinates.h0_subntt_idx = 0; // not relevant for layer 2
        int log_nof_blocks = NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0] +
                             NttCpu<S, E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1];
        for (ntt_task_cordinates.h0_block_idx = 0; ntt_task_cordinates.h0_block_idx < (1 << log_nof_blocks);
             ntt_task_cordinates.h0_block_idx++) {
          ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, false);
        }
      }
    }
    // int nof_h0_subntts = (nof_h0_layers == 1) ? (1 << NttCpu<S,
    // E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]) :
    //                      (nof_h0_layers == 2) ? (1 << NttCpu<S,
    //                      E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]) : 1;
    // int nof_h0_blocks  = (nof_h0_layers != 3) ? (1 << NttCpu<S,
    // E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][2]) : (1 << (NttCpu<S,
    // E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][0]+NttCpu<S,
    // E>::ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][1]));
    if (nof_h0_layers > 1) {
      ntt_task_cordinates = {ntt_task_cordinates.h1_layer_idx, ntt_task_cordinates.h1_subntt_idx, nof_h0_layers, 0, 0};
      ntt_tasks_manager.push_task(this, input, ntt_task_cordinates, true); // reorder=true
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Executes the NTT on a sub-NTT at the h0 hierarchy level.
   *
   * This function applies the NTT on a sub-NTT specified by the task coordinates at the h0 level.
   * h0_dit_ntt transforming the data from the bit-reversed order (R) to natural order (N) so
   * this function first reorders the input elements by bit-reversing their indices, then performs the DIT NTT.
   * If further refactoring is required, the output is processed to prepare it for the next layer.
   *
   * @param input The input array of elements to be transformed.
   * @param ntt_task_cordinates The coordinates specifying the sub-NTT within the NTT hierarchy.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::h0_cpu_ntt(E* input, NttTaskCordinates ntt_task_cordinates)
  {
    // std::cout << "h0_cpu_ntt     - h1_subntt_idx:\t" << ntt_task_cordinates.h1_subntt_idx<< std::endl;
    const uint64_t subntt_size =
      (1 << NttCpu<S, E>::ntt_sub_logn
              .h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx]);
    uint64_t original_size = (this->ntt_sub_logn.size);
    const uint64_t total_memory_size = original_size * this->config.batch_size;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    int offset = this->config.columns_batch ? this->config.batch_size : 1;
    E* current_input =
      input + offset * (ntt_task_cordinates.h1_subntt_idx
                        << NttCpu<S, E>::ntt_sub_logn
                             .h1_layers_sub_logn[ntt_task_cordinates.h1_layer_idx]); // input + subntt_idx * subntt_size

    this->reorder_by_bit_reverse(
      ntt_task_cordinates, current_input,
      false); // TODO - check if access the fixed indexes instead of reordering may be more efficient?

    // NTT/INTT
    this->h0_dit_ntt(current_input, ntt_task_cordinates, twiddles); // R --> N

    if (
      ntt_task_cordinates.h0_layer_idx != 2 &&
      this->ntt_sub_logn.h0_layers_sub_logn[ntt_task_cordinates.h1_layer_idx][ntt_task_cordinates.h0_layer_idx + 1] !=
        0) {
      this->refactor_output_h0(input, ntt_task_cordinates, twiddles);
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
   * @param h1_layer_idx The index of the current h1 layer being processed.
   * @return eIcicleError Returns `SUCCESS` if all tasks are successfully handled.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::handle_pushed_tasks(
    TasksManager<NttTask<S, E>>* tasks_manager, NttTasksManager<S, E>& ntt_tasks_manager, int h1_layer_idx)
  {
    NttTask<S, E>* task_slot = nullptr;
    NttTaskParams<S, E> params;

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
  template <>
  struct hash<ntt_cpu::NttTaskCordinates> {
    std::size_t operator()(const ntt_cpu::NttTaskCordinates& key) const
    {
      // Combine hash values of the members using a simple hash combiner
      return ((std::hash<int>()(key.h1_layer_idx) ^ (std::hash<int>()(key.h1_subntt_idx) << 1)) >> 1) ^
             (std::hash<int>()(key.h0_layer_idx) << 1) ^ (std::hash<int>()(key.h0_block_idx) << 1) ^
             (std::hash<int>()(key.h0_subntt_idx) << 1);
    }
  };
} // namespace std
