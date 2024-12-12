#pragma once
#include "icicle/utils/log.h"
#include "ntt_task.h"
#include <cstdint>

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
   * @method TasksDependenciesCounters(NttSubHierarchies ntt_sub_hierarchies, uint32_t hierarchy_1_layer_idx)
   * Constructor that initializes the counters based on NTT structure.
   * @method bool decrement_counter(NttTaskCoordinates ntt_task_coordinates) Decrements the counter for a given task and
   * returns true if the task is ready to execute.
   * @method uint32_t get_dependent_subntt_count(uint32_t hierarchy_0_layer_idx) Returns the number of counters pointing
   * to the given hierarchy_0 layer.
   * @method uint32_t get_nof_hierarchy_0_layers() Returns the number of hierarchy_0 layers in the current hierarchy_1
   * layer.
   */
  class TasksDependenciesCounters
  {
  public:
    // Constructor that initializes the counters
    TasksDependenciesCounters(const NttSubHierarchies& ntt_sub_hierarchies, uint32_t hierarchy_1_layer_idx, const uint32_t nof_elems_per_cacheline);

    // Function to decrement the counter for a given task and check if it is ready to execute. if so, return true
    bool decrement_counter(NttTaskCoordinates ntt_task_coordinates, const uint32_t nof_elems_per_cacheline);
    uint32_t get_dependent_subntt_count(uint32_t hierarchy_0_layer_idx)
    {
      return dependent_subntt_count[hierarchy_0_layer_idx];
    }

  private:
    uint32_t hierarchy_1_layer_idx;               // Index of the current hierarchy 1 layer.
    uint32_t nof_hierarchy_0_layers;              // Number of hierarchy 0 layers in the current hierarchy 1 layer.
    std::vector<uint32_t> dependent_subntt_count; // Number of subntt that are getting available together when a group
                                                  // of hierarchy_0_subntts from previous layer are done

    std::vector<std::vector<std::vector<uint32_t>>>
      hierarchy_0_counters; // 3D vector of counters for groups of sub-NTTs in hierarchy 0 layers:
                            // hierarchy_0_counters[hierarchy_1_subntt_idx][hierarchy_0_layer_idx][hierarchy_0_counter_idx]

    // One counter for each hierarchy_1_subntt to signal the end of the hierarchy_1_subntt. each hierarchy_0_subntt of
    // last hierarchy_0_layer will decrement this counter when it finishes and when it reaches 0, the hierarchy_1_subntt
    // is ready to reorder
    std::vector<uint32_t> hierarchy_1_counters; // hierarchy_1_counters[hierarchy_1_subntt_idx]
  };

  /**
   * @brief Manages tasks for the NTT computation, handling task scheduling and dependency management.
   *
   * The NttTasksManager is responsible for adding tasks, updating task dependencies,
   * and determining the readiness of tasks for execution. This class ensures that
   * tasks are executed in the correct order based on their dependencies within the NTT hierarchy.
   *
   */
  template <typename S = scalar_t, typename E = scalar_t>
  class NttTasksManager
  {
  public:
    NttTasksManager(const NttSubHierarchies& ntt_sub_hierarchies, const uint32_t logn, uint32_t nof_elems_per_cacheline);

    // Add a new task to the ntt_task_manager
    eIcicleError push_task(const NttTaskCoordinates& ntt_task_coordinates);

    // Set a task as completed and update dependencies
    bool handle_completed(NttTask<S, E>* completed_task, uint32_t nof_subntts_l1);
    NttTaskCoordinates* get_slot_for_next_task_coordinates();

    bool tasks_to_do() const;
    bool available_tasks() const;
    NttTaskCoordinates* get_available_task();
    uint32_t nof_pending_tasks = 0; // the current count of tasks that are pending execution

  private:
    const uint32_t logn;                             // log of the NTT size
    const NttSubHierarchies& ntt_sub_hierarchies;    // Reference to NttSubHierarchies
    uint32_t nof_elems_per_cacheline;                // Number of elements per cacheline
    std::vector<TasksDependenciesCounters> counters; // Dependencies counters by layer
    std::vector<NttTaskCoordinates> task_buffer;     // Buffer holding task coordinates for pending tasks
    size_t head; // Head index for the task buffer (used in circular buffer implementation)
    size_t tail; // Tail index for the task buffer (used in circular buffer implementation)

    bool is_full() const;
    bool is_empty() const;
    void increment(size_t& index);
  };

  //////////////////////////// TasksDependenciesCounters Implementation ////////////////////////////

  /**
   * @brief Constructs a TasksDependenciesCounters instance with specified NTT sub-logarithms and hierarchy layer index.
   *
   * Initializes dependency counters based on the provided NTT structure and hierarchy layer. It sets up
   * counters for each sub-NTT in hierarchy 1 and initializes counters for hierarchy 0 layers.
   *
   * @param ntt_sub_hierarchies The structure containing logarithmic sizes of sub-NTTs.
   * @param hierarchy_1_layer_idx The index of the current hierarchy 1 layer.
   */
  TasksDependenciesCounters::TasksDependenciesCounters(
    const NttSubHierarchies& ntt_sub_hierarchies, uint32_t hierarchy_1_layer_idx, const uint32_t nof_elems_per_cacheline)
      : hierarchy_0_counters(
          1 << ntt_sub_hierarchies.hierarchy_1_layers_sub_logn
                 [1 - hierarchy_1_layer_idx]), // nof_hierarchy_1_subntts =
                                               // hierarchy_1_layers_sub_logn[1-hierarchy_1_layer_idx].
        hierarchy_1_counters(1 << ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1 - hierarchy_1_layer_idx])
  {
    nof_hierarchy_0_layers = ntt_sub_hierarchies.hier0_layer_counts_in_hier1[hierarchy_1_layer_idx];
    dependent_subntt_count.resize(nof_hierarchy_0_layers);
    dependent_subntt_count[0] = 1;
    uint32_t l1_counter_size;
    uint32_t l2_counter_size;
    uint32_t l1_nof_counters;
    uint32_t l2_nof_counters;
    if (nof_hierarchy_0_layers == 2) {
      // Initialize counters for layer 1 - N2 counters initialized with N1.
      dependent_subntt_count[1] = 1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0]; //when the counter is 0 this is the number of subntts that are ready to execute
      l1_nof_counters = 1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2]; // number of counters in the layer
      l1_counter_size = (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1])/nof_elems_per_cacheline; // number of tasks to be done before the dependent tasks are ready
    }
    if (nof_hierarchy_0_layers == 3) {
      // Initialize counters for layer 1 - N2 counters initialized with N1.
      dependent_subntt_count[1] = 1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0];
      l1_nof_counters = (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2])/nof_elems_per_cacheline;
      l1_counter_size = (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1]);
      // Initialize counters for layer 2 - N0 counters initialized with N2.
      dependent_subntt_count[2] = 1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1];
      l2_nof_counters = 1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0];
      l2_counter_size = (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2])/nof_elems_per_cacheline;
    }

    for (uint32_t hierarchy_1_subntt_idx = 0;
         hierarchy_1_subntt_idx < (1 << ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1 - hierarchy_1_layer_idx]);
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
        nof_hierarchy_0_layers == 3 ? (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0]) *
                                        (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1])
        : nof_hierarchy_0_layers == 2 ? (1 << ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0])
                                      : 0;
      hierarchy_1_counters[hierarchy_1_subntt_idx] = hierarchy_1_counter_size;
    }
  }

  /**
   * @brief Decrements the dependency counter for a given task and checks if the dependent task is ready to execute.
   *
   * This function decrements the counter associated with a task in hierarchy 0 or the global counter in hierarchy 1.
   * If the counter reaches zero, it indicates that the dependent task is now ready to be executed.
   *
   * @param task_c The coordinates of the task whose counter is to be decremented.
   * @return `true` if the dependent task is ready to execute, `false` otherwise.
   */
  bool TasksDependenciesCounters::decrement_counter(NttTaskCoordinates task_c, const uint32_t nof_elems_per_cacheline)
  {
    if (nof_hierarchy_0_layers == 1) { return false; }
    if (task_c.hierarchy_0_layer_idx < nof_hierarchy_0_layers - 1) {
      // Extract the coordinates from the task
      uint32_t counter_group_idx =
        task_c.hierarchy_0_layer_idx == 0 ? (nof_hierarchy_0_layers ==2 ? task_c.hierarchy_0_block_idx : task_c.hierarchy_0_block_idx / nof_elems_per_cacheline):
      /*task_c.hierarchy_0_layer_idx==1*/   task_c.hierarchy_0_subntt_idx;

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
  NttTasksManager<S, E>::NttTasksManager(const NttSubHierarchies& ntt_sub_hierarchies, uint32_t logn,const uint32_t nof_elems_per_cacheline)
      : logn(logn), ntt_sub_hierarchies(ntt_sub_hierarchies), nof_elems_per_cacheline(nof_elems_per_cacheline),
        counters(logn > HIERARCHY_1 ? 2 : 1, TasksDependenciesCounters(ntt_sub_hierarchies, 0, nof_elems_per_cacheline)),
        task_buffer(1 << (logn)), // Pre-allocate buffer
        head(0), tail(0), nof_pending_tasks(0)

  {
    if (logn > HIERARCHY_1) { counters[1] = TasksDependenciesCounters(ntt_sub_hierarchies, 1, nof_elems_per_cacheline); }
  }

  /**
   * @brief Checks if the task buffer is full.
   *
   * Determines whether the task buffer has reached its maximum capacity.
   *
   * @return `true` if the buffer is full, `false` otherwise.
   */
  template <typename S, typename E>
  bool NttTasksManager<S, E>::is_full() const
  {
    return (tail + 1) % (1 << (logn)) == head;
  }

  /**
   * @brief Checks if the task buffer is empty.
   *
   * Determines whether there are no tasks in the task buffer and no pending tasks.
   *
   * @return `true` if the buffer is empty and there are no pending tasks, `false` otherwise.
   */
  template <typename S, typename E>
  bool NttTasksManager<S, E>::is_empty() const
  {
    return head == tail && nof_pending_tasks == 0;
  }

  /**
   * @brief Increments a buffer index in a circular manner.
   *
   * Advances the given index to the next position in the circular buffer, wrapping around if necessary.
   *
   * @param index Reference to the index to be incremented.
   */
  template <typename S, typename E>
  void NttTasksManager<S, E>::increment(size_t& index)
  {
    index = (index + 1) % (1 << (logn));
  }

  /**
   * @brief Adds a new task to the task manager.
   *
   * Inserts a new task into the task buffer if there is available space. This task will be managed
   * by the task manager, which will handle its execution based on dependency resolution.
   *
   * @param ntt_task_coordinates Task coordinates specifying the task's position in the hierarchy.
   * @return `eIcicleError::SUCCESS` if the task was successfully added, otherwise an appropriate error code.
   */
  template <typename S, typename E>
  eIcicleError NttTasksManager<S, E>::push_task(const NttTaskCoordinates& ntt_task_coordinates)
  {
    if (is_full()) { return eIcicleError::OUT_OF_MEMORY; }
    task_buffer[tail] = ntt_task_coordinates;
    increment(tail);
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Retrieves a pointer to a slot for the next task coordinates.
   *
   * Provides access to a slot in the task buffer where new task coordinates can be assigned.
   * @return Pointer to `NttTaskCoordinates` if a slot is available, `nullptr` otherwise.
   */
  template <typename S, typename E>
  NttTaskCoordinates* NttTasksManager<S, E>::get_slot_for_next_task_coordinates()
  {
    if (is_full()) { return nullptr; }
    NttTaskCoordinates* task = &task_buffer[tail];
    increment(tail);
    return task;
  }

  /**
   * @brief Retrieves the next available task ready for execution.
   *
   * Fetches the next task from the available task buffer that is ready to be executed based on dependency
   * resolution. If no tasks are available, returns `nullptr`.
   *
   * @return Pointer to `NttTaskCoordinates` of the available task, or `nullptr` if none are available.
   */
  template <typename S, typename E>
  NttTaskCoordinates* NttTasksManager<S, E>::get_available_task()
  {
    if (head == tail) {
      // No available tasks
      return nullptr;
    }
    NttTaskCoordinates* task = &task_buffer[head];
    increment(head);
    // ICICLE_LOG_DEBUG << "Popping task: " 
    //                  << task->hierarchy_1_layer_idx << ", "
    //                  << task->hierarchy_1_subntt_idx << ", " 
    //                  << task->hierarchy_0_layer_idx << ", " 
    //                  << task->hierarchy_0_block_idx << ", "
    //                  << task->hierarchy_0_subntt_idx;
    // ICICLE_LOG_DEBUG << "head: " << head;
    return task;
  }

  /**
   * @brief Checks if there are tasks remaining to be processed.
   *
   * Determines whether there are any tasks left to execute or pending dependencies.
   *
   * @return `true` if there are tasks to do, `false` otherwise.
   */
  template <typename S, typename E>
  bool NttTasksManager<S, E>::tasks_to_do() const
  {
    ICICLE_LOG_DEBUG << "nof_pending_tasks: " << nof_pending_tasks;
    ICICLE_LOG_DEBUG << "head: " << head;
    ICICLE_LOG_DEBUG << "tail: " << tail;
    return head != tail || nof_pending_tasks != 0;
  }

  /**
   * @brief Checks if there are available tasks ready for execution.
   *
   * Determines whether there are any tasks in the buffer that are ready to be executed.
   *
   * @return `true` if there are available tasks, `false` otherwise.
   */
  template <typename S, typename E>
  bool NttTasksManager<S, E>::available_tasks() const
  {
    return head != tail;
  }

  /**
   * @brief Marks a task as completed and updates dependencies.
   *
   * This function should be called when a task has finished execution. It decrements the relevant
   * dependency counters and, if dependencies are resolved, dispatches dependent tasks for execution or adds them to the
   * task buffer.
   *
   * @param completed_task Pointer to the completed task.
   * @param nof_subntts_l1 Number of sub-NTTs in the second layer of hierarchy 1.
   * @return `true` if a dependent task was dispatched as a result of this completion, `false` otherwise.
   */
  template <typename S, typename E>
  bool NttTasksManager<S, E>::handle_completed(NttTask<S, E>* completed_task, uint32_t nof_subntts_l1)
  {
    bool task_dispatched = false;
    NttTaskCoordinates task_c = *completed_task->get_coordinates();
    uint32_t nof_hierarchy_0_layers = ntt_sub_hierarchies.hier0_layer_counts_in_hier1[task_c.hierarchy_1_layer_idx];
    // Update dependencies in counters
    if (counters[task_c.hierarchy_1_layer_idx].decrement_counter(task_c, nof_elems_per_cacheline)) {
      if (task_c.hierarchy_0_layer_idx < nof_hierarchy_0_layers - 1) {
        NttTaskCoordinates* next_task_c_ptr = nullptr;
        uint32_t nof_new_ready_tasks =
          (task_c.hierarchy_0_layer_idx == nof_hierarchy_0_layers - 1)
            ? 1
            : counters[task_c.hierarchy_1_layer_idx].get_dependent_subntt_count(task_c.hierarchy_0_layer_idx + 1);
        uint32_t stride = nof_subntts_l1 / nof_new_ready_tasks;

        for (uint32_t i = 0; i < nof_new_ready_tasks; i++) {
          next_task_c_ptr = get_slot_for_next_task_coordinates();
          next_task_c_ptr->hierarchy_1_layer_idx = task_c.hierarchy_1_layer_idx;
          next_task_c_ptr->hierarchy_1_subntt_idx = task_c.hierarchy_1_subntt_idx;
          next_task_c_ptr->hierarchy_0_layer_idx = task_c.hierarchy_0_layer_idx + 1;
          next_task_c_ptr->hierarchy_0_block_idx = (task_c.hierarchy_0_layer_idx == 0)
                                                     ? task_c.hierarchy_0_block_idx
                                                     : task_c.hierarchy_0_subntt_idx + stride * i;
          next_task_c_ptr->hierarchy_0_subntt_idx = (task_c.hierarchy_0_layer_idx == 0) ? i : 0;
          ICICLE_LOG_DEBUG << "Pushing new task: " 
                           << next_task_c_ptr->hierarchy_1_layer_idx << ", "
                           << next_task_c_ptr->hierarchy_1_subntt_idx << ", " 
                           << next_task_c_ptr->hierarchy_0_layer_idx << ", " 
                           << next_task_c_ptr->hierarchy_0_block_idx << ", "
                           << next_task_c_ptr->hierarchy_0_subntt_idx;
          if (i == 0) {
            completed_task->set_coordinates(get_available_task());
            completed_task->dispatch();
            task_dispatched = true;
          }
          nof_pending_tasks--;
        }
      } else {
        // Reorder the output
        NttTaskCoordinates* next_task_c_ptr = nullptr;
        next_task_c_ptr = get_slot_for_next_task_coordinates();
        next_task_c_ptr->hierarchy_1_layer_idx = task_c.hierarchy_1_layer_idx;
        next_task_c_ptr->hierarchy_1_subntt_idx = task_c.hierarchy_1_subntt_idx;
        next_task_c_ptr->hierarchy_0_layer_idx = nof_hierarchy_0_layers;
        next_task_c_ptr->hierarchy_0_block_idx = 0;
        next_task_c_ptr->hierarchy_0_subntt_idx = 0;
        next_task_c_ptr->reorder = true;
        ICICLE_LOG_DEBUG << "Pushing task: " 
                  << next_task_c_ptr->hierarchy_1_layer_idx << ", "
                  << next_task_c_ptr->hierarchy_1_subntt_idx << ", " 
                  << next_task_c_ptr->hierarchy_0_layer_idx << ", " 
                  << next_task_c_ptr->hierarchy_0_block_idx << ", "
                  << next_task_c_ptr->hierarchy_0_subntt_idx << ", "
                  << "Reorder";
        completed_task->set_coordinates(get_available_task());
        completed_task->dispatch();
        task_dispatched = true;
        nof_pending_tasks--;
      }
    }
    return task_dispatched;
  }
} // namespace ntt_cpu
