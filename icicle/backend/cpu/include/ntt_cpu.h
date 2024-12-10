#pragma once
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/utils/cacheline_size.h"
#include "ntt_tasks_manager.h"
#include "ntt_utils.h"
// #include <_types/_uint32_t.h>
#include <cstdint>
#include <deque>

using namespace field_config;
using namespace icicle;

namespace ntt_cpu {
  /**
   * @brief Constructs an `NttCpu` instance with the specified parameters.
   *
   * Initializes the NTT data structures, task managers, and input/output buffers.
   *
   * @param logn       The log of the size of the NTT.
   * @param direction  The direction of the NTT computation, either forward or inverse.
   * @param config     Configuration settings for the NTT, including batch size and ordering.
   * @param input      Pointer to the input data array.
   * @param output     Pointer to the output data array where results will be stored.
   */
  template <typename S = scalar_t, typename E = scalar_t>
  class NttCpu
  {
  public:
    NttCpu(uint32_t logn, NTTDir direction, const NTTConfig<S>& config, const E* input, E* output);

    eIcicleError run();

  private:
    const E* input;
    NttData<S, E> ntt_data;

    // Parallel-specific members
    std::optional<NttTasksManager<S, E>> ntt_tasks_manager;
    std::unique_ptr<TasksManager<NttTask<S, E>>> tasks_manager;

    bool compute_if_is_parallel(uint32_t logn, const NTTConfig<S>& config);
    void coset_mul();
    void reorder_by_bit_reverse();
    void copy_and_reorder_if_needed(const E* input, E* output);

    // Parallel-specific methods
    eIcicleError hierarchy1_push_tasks(uint32_t hierarchy_1_layer_idx, uint32_t hierarchy_1_subntt_idx);
    eIcicleError handle_pushed_tasks(uint32_t hierarchy_1_layer_idx);
    void hierarchy_1_reorder();
    eIcicleError reorder_output();

  }; // class NttCpu

  //////////////////////////// NttCpu Implementation ////////////////////////////

  /*
   * @brief Constructs an `NttCpu` instance
   */
  template <typename S, typename E>
  NttCpu<S, E>::NttCpu(uint32_t logn, NTTDir direction, const NTTConfig<S>& config, const E* input, E* output)
      : input(input), ntt_data(logn, output, config, direction, compute_if_is_parallel(logn, config), get_cache_line_size()/ sizeof(E)), 
        ntt_tasks_manager(
          ntt_data.is_parallel ? std::optional<NttTasksManager<S, E>>(std::in_place, ntt_data.ntt_sub_hierarchies, logn, ntt_data.nof_elems_per_cacheline)
                               : std::nullopt),
        tasks_manager(
          ntt_data.is_parallel ? std::make_unique<TasksManager<NttTask<S, E>>>(std::thread::hardware_concurrency() - 2)
                               : nullptr)
  {
        ICICLE_LOG_DEBUG << "Number of elements per cacheline: " << ntt_data.nof_elems_per_cacheline;
        ICICLE_LOG_DEBUG << "NOF WORKERS " << std::thread::hardware_concurrency() - 2;
  }

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::run()
  {
    copy_and_reorder_if_needed(input, ntt_data.elements);
    if (ntt_data.is_parallel) {
      if (ntt_data.logn > HIERARCHY_1) {
        for (uint32_t hierarchy_1_layer_idx = 0; hierarchy_1_layer_idx < 2; hierarchy_1_layer_idx++) {
          const uint32_t sunbtt_plus_batch_logn =
            ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[hierarchy_1_layer_idx] +
            uint32_t(log2(ntt_data.config.batch_size));
          const uint32_t log_nof_hierarchy_1_subntts_todo_in_parallel =
            sunbtt_plus_batch_logn < HIERARCHY_1 ? HIERARCHY_1 - sunbtt_plus_batch_logn : 0;
          const uint32_t nof_hierarchy_1_subntts_todo_in_parallel = 1 << log_nof_hierarchy_1_subntts_todo_in_parallel;
          const uint32_t log_nof_subntts_chunks =
            ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1 - hierarchy_1_layer_idx] -
            log_nof_hierarchy_1_subntts_todo_in_parallel;
          const uint32_t nof_subntts_chunks = 1 << log_nof_subntts_chunks;
          for (uint32_t hierarchy_1_subntts_chunck_idx = 0; hierarchy_1_subntts_chunck_idx < nof_subntts_chunks;
               hierarchy_1_subntts_chunck_idx++) {
            for (uint32_t hierarchy_1_subntt_idx_in_chunck = 0;
                 hierarchy_1_subntt_idx_in_chunck < nof_hierarchy_1_subntts_todo_in_parallel;
                 hierarchy_1_subntt_idx_in_chunck++) {
              hierarchy1_push_tasks(
                hierarchy_1_layer_idx, hierarchy_1_subntts_chunck_idx * nof_hierarchy_1_subntts_todo_in_parallel +
                                         hierarchy_1_subntt_idx_in_chunck);
            }
            handle_pushed_tasks(hierarchy_1_layer_idx);
          }
          if (hierarchy_1_layer_idx == 0) { hierarchy_1_reorder(); }
        }
        reorder_output();
      } else {
        hierarchy1_push_tasks(0, 0);
        handle_pushed_tasks(0);
      }
    } else {
      if (ntt_data.direction == NTTDir::kForward && ntt_data.config.coset_gen != S::one()) { coset_mul(); }
      NttTask<S, E> task;
      task.set_data(ntt_data);
      task.execute();
    }
    if (ntt_data.direction == NTTDir::kInverse && ntt_data.config.coset_gen != S::one()) { coset_mul(); }

    if (ntt_data.config.ordering == Ordering::kNR || ntt_data.config.ordering == Ordering::kRR) {
      reorder_by_bit_reverse();
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Copies the input data to the output buffer with necessary reordering.
   *
   * Depending on the size of the input data and the specified ordering configuration,
   * this function performs bit-reversal and/or custom reordering logic. If no reordering
   * is needed, the data is copied as-is. It handles scenarios where input and output
   * buffers may overlap by using temporary storage if necessary.
   *
   * @param input  Pointer to the input array to be copied and reordered.
   * @param output Pointer to the output array where the data will be stored.
   * @return E*    Pointer to the output array after copying and reordering.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::copy_and_reorder_if_needed(const E* input, E* output)
  {
    const uint64_t total_memory_size = ntt_data.size * ntt_data.config.batch_size;
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const uint32_t logn = static_cast<uint32_t>(std::log2(ntt_data.size));
    const bool bit_rev = ntt_data.config.ordering == Ordering::kRN || ntt_data.config.ordering == Ordering::kRR;

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
      uint32_t cur_ntt_log_size = ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[0];
      uint32_t next_ntt_log_size = ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1];

      for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
        const E* input_batch = ntt_data.config.columns_batch ? (input + batch) : (input + batch * ntt_data.size);
        E* output_batch = ntt_data.config.columns_batch ? (temp_output + batch) : (temp_output + batch * ntt_data.size);

        for (uint64_t i = 0; i < ntt_data.size; ++i) {
          uint32_t subntt_idx = i >> cur_ntt_log_size;
          uint32_t element = i & ((1 << cur_ntt_log_size) - 1);
          uint64_t new_idx = bit_rev ? bit_reverse(subntt_idx + (element << next_ntt_log_size), logn)
                                     : subntt_idx + (element << next_ntt_log_size);
          output_batch[stride * i] = input_batch[stride * new_idx];
        }
      }

    } else if (bit_rev) {
      // Only bit-reverse reordering needed
      for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
        const E* input_batch = ntt_data.config.columns_batch ? (input + batch) : (input + batch * ntt_data.size);
        E* output_batch = ntt_data.config.columns_batch ? (temp_output + batch) : (temp_output + batch * ntt_data.size);

        for (uint64_t i = 0; i < ntt_data.size; ++i) {
          // uint64_t rev = NttUtils<S, E>::bit_reverse(i, logn);
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
  }

  /**
   * @brief Applies coset multiplication to the elements.
   *
   * This function performs coset multiplication on the `elements` array based on
   * the coset generator specified in the configuration. It adjusts indices according
   * to any prior reordering to ensure correct multiplication.
   * Used in the inverse NTT.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::coset_mul()
  {
    uint32_t batch_stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const bool needs_reorder_input = ntt_data.direction == NTTDir::kForward && (ntt_data.logn > HIERARCHY_1);
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;

      for (uint64_t i = 1; i < ntt_data.size; ++i) {
        uint64_t idx = i;

        // Adjust the index if reorder logic was applied on the input
        if (needs_reorder_input) {
          uint32_t cur_ntt_log_size = ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[0];
          uint32_t next_ntt_log_size = ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1];
          uint32_t subntt_idx = i >> cur_ntt_log_size;
          uint32_t element = i & ((1 << cur_ntt_log_size) - 1);
          idx = subntt_idx + (element << next_ntt_log_size);
        }

        // Apply coset multiplication based on the available coset information
        if (ntt_data.arbitrary_coset) {
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * ntt_data.arbitrary_coset[idx];
        } else {
          uint32_t twiddle_idx = ntt_data.coset_stride * idx;
          twiddle_idx = ntt_data.direction == NTTDir::kForward
                          ? twiddle_idx
                          : CpuNttDomain<S>::s_ntt_domain.get_max_size() - twiddle_idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * twiddles[twiddle_idx];
        }
      }
    }
  }

  /**
   * @brief Reorders elements between layers of hierarchy 1, based on sub-NTT structure.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::hierarchy_1_reorder()
  {
    const uint32_t sntt_size = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1];
    const uint32_t nof_sntts = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[0];
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const uint64_t temp_elements_size = ntt_data.size * ntt_data.config.batch_size;

    auto temp_elements = std::make_unique<E[]>(temp_elements_size);
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* cur_layer_output =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;
      E* cur_temp_elements =
        ntt_data.config.columns_batch ? temp_elements.get() + batch : temp_elements.get() + batch * ntt_data.size;
      for (uint32_t sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (uint32_t elem = 0; elem < sntt_size; elem++) {
          cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
            cur_layer_output[stride * (elem * nof_sntts + sntt_idx)];
        }
      }
    }
    std::copy(temp_elements.get(), temp_elements.get() + temp_elements_size, ntt_data.elements);
  }

  /**
   * @brief Reorders the output data after completing hierarchy 1 computations.
   *
   * This function rearranges the output elements based on the hierarchy levels and
   * sub-NTT structures so that the final output is in the correct order.
   *
   * @return eIcicleError Returns SUCCESS if reordering is successful, or an error code otherwise.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::reorder_output()
  {
    uint32_t columns_batch_reps = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    uint32_t rows_batch_reps = ntt_data.config.columns_batch ? 1 : ntt_data.config.batch_size;
    uint32_t s0 = ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[0];
    uint32_t s1 = ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1];
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t row_batch = 0; row_batch < rows_batch_reps;
         ++row_batch) { // if columns_batch=false, then elements pointer is shifted by batch*size
      E* elements = ntt_data.elements + row_batch * ntt_data.size;
      uint64_t temp_output_size =
        ntt_data.config.columns_batch ? ntt_data.size * ntt_data.config.batch_size : ntt_data.size;
      auto temp_output = std::make_unique<E[]>(temp_output_size);
      uint64_t new_idx = 0;
      uint32_t subntt_idx;
      uint32_t element;
      for (uint32_t col_batch = 0; col_batch < columns_batch_reps; ++col_batch) {
        E* current_elements =
          ntt_data.config.columns_batch
            ? elements + col_batch
            : elements; // if columns_batch=true, then elements pointer is shifted by 1 for each batch
        E* current_temp_output = ntt_data.config.columns_batch ? temp_output.get() + col_batch : temp_output.get();
        for (uint64_t i = 0; i < ntt_data.size; i++) {
          subntt_idx = i >> s1;
          element = i & ((1 << s1) - 1);
          new_idx = subntt_idx + (element << s0);
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
      std::copy(
        temp_output.get(), temp_output.get() + temp_output_size,
        elements); // columns_batch=false: for each row in the batch, copy the reordered elements back to the elements
                   // array
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Reorders the elements by performing a bit-reversal permutation.
   *
   * When the configuration specifies bit-reversed ordering (RN or RR), this function
   * rearranges the elements accordingly. It swaps elements whose indices are bitwise
   * reversed with respect to each other, ensuring the output meets the required ordering.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::reorder_by_bit_reverse()
  {
    uint64_t subntt_size = (ntt_data.size);
    uint32_t subntt_log_size = (ntt_data.logn);
    uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;
      uint64_t rev;
      for (uint64_t i = 0; i < subntt_size; ++i) {
        rev = bit_reverse(i, subntt_log_size);
        if (i < rev) { std::swap(current_elements[stride * i], current_elements[stride * rev]); }
      }
    }
  }

  /**
   * @brief Schedules tasks for the first hierarchy layer of the NTT computation.
   *
   * This function organizes and pushes tasks corresponding to a specific hierarchy 1 layer
   * and sub-NTT index into the task manager. It calculates the number of blocks and sub-NTTs
   * based on the layer indices and logs, then schedules tasks accordingly. If multiple hierarchy 0
   * layers are involved, it also schedules a reorder task after processing.
   *
   * @param hierarchy_1_layer_idx    Index of the current hierarchy 1 layer.
   * @param hierarchy_1_subntt_idx   Index of the sub-NTT within the hierarchy 1 layer.
   * @return eIcicleError            Returns SUCCESS if tasks are successfully scheduled, or an error code otherwise.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::hierarchy1_push_tasks(uint32_t hierarchy_1_layer_idx, uint32_t hierarchy_1_subntt_idx)
  {
    if (!ntt_tasks_manager) {
      return eIcicleError::UNKNOWN_ERROR; // Handle case where no task manager is available
    }

    uint32_t nof_hierarchy_0_layers =
      (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2] != 0)   ? 3
      : (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1] != 0) ? 2
                                                                                                  : 1;
    uint64_t nof_blocks;
    uint64_t nof_subntts;
    for (uint32_t hierarchy_0_layer_idx = 0; hierarchy_0_layer_idx < nof_hierarchy_0_layers; hierarchy_0_layer_idx++) {
      if (hierarchy_0_layer_idx == 0) {
        nof_blocks = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2];
        nof_subntts = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1];
      } else if (hierarchy_0_layer_idx == 1) {
        nof_blocks = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2];
        nof_subntts = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0];
      } else {
        nof_blocks = 1
                     << (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0] +
                         ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1]);
        nof_subntts = 1;
      }
      for (uint32_t hierarchy_0_block_idx = 0; hierarchy_0_block_idx < (nof_blocks); hierarchy_0_block_idx++) {
        // for (uint32_t hierarchy_0_subntt_idx = 0; hierarchy_0_subntt_idx < (nof_subntts); hierarchy_0_subntt_idx++) {
        for (uint32_t hierarchy_0_subntt_idx = 0; hierarchy_0_subntt_idx < (nof_subntts); hierarchy_0_subntt_idx+=ntt_data.nof_elems_per_cacheline) {
          if (hierarchy_0_layer_idx == 0) {
            NttTaskCoordinates* ntt_task_coordinates = ntt_tasks_manager->get_slot_for_next_task_coordinates();
            ntt_task_coordinates->hierarchy_1_layer_idx = hierarchy_1_layer_idx;
            ntt_task_coordinates->hierarchy_1_subntt_idx = hierarchy_1_subntt_idx;
            ntt_task_coordinates->hierarchy_0_layer_idx = hierarchy_0_layer_idx;
            ntt_task_coordinates->hierarchy_0_block_idx = hierarchy_0_block_idx;
            ntt_task_coordinates->hierarchy_0_subntt_idx = hierarchy_0_subntt_idx;
            ntt_task_coordinates->reorder = false;
            ICICLE_LOG_DEBUG << "Pushing task: " << ntt_task_coordinates->hierarchy_1_layer_idx << ", "
                             << ntt_task_coordinates->hierarchy_1_subntt_idx << ", "
                             << ntt_task_coordinates->hierarchy_0_layer_idx << ", "
                             << ntt_task_coordinates->hierarchy_0_block_idx << ", "
                             << ntt_task_coordinates->hierarchy_0_subntt_idx;
          } else {
            ntt_tasks_manager->nof_pending_tasks+=ntt_data.nof_elems_per_cacheline;
            ICICLE_LOG_DEBUG << "no pending tasks: " << ntt_tasks_manager->nof_pending_tasks;
          }
        }
      }
    }
    if (nof_hierarchy_0_layers > 1) { // all ntt tasks in hierarchy 1 are pushed, now push reorder task so that the data
                                      // is in the correct order for the next hierarchy 1 layer
      ntt_tasks_manager->nof_pending_tasks++;
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Manages the execution and completion of scheduled tasks.
   *
   * This function handles the lifecycle of tasks at a given hierarchy level. It retrieves
   * available tasks from the task manager, dispatches them for execution, and processes
   * their completion. The function ensures that all tasks are executed and dependencies
   * are correctly managed, including idle states for waiting tasks.
   *
   * @param hierarchy_1_layer_idx Index of the current hierarchy 1 layer being processed.
   * @return eIcicleError         Returns SUCCESS if all tasks are successfully handled, or an error code otherwise.
   */
  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::handle_pushed_tasks(uint32_t hierarchy_1_layer_idx)
  {
    if (!ntt_tasks_manager) { return eIcicleError::UNKNOWN_ERROR; }

    NttTask<S, E>* task_slot = nullptr;
    // std::deque<NttTaskCoordinates> completed_tasks_list;

    uint32_t nof_subntts_l1 = 1
                              << ((ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0]) +
                                  (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1]));
    while (ntt_tasks_manager->tasks_to_do()) {
      // There are tasks that are available or waiting

      if (ntt_tasks_manager->available_tasks()) {
        // Task is available to dispatch
        task_slot = tasks_manager->get_idle_or_completed_task();
        if (task_slot->is_completed()) {
          if (ntt_tasks_manager->handle_completed(task_slot, nof_subntts_l1)) { continue; }
        } else {
          task_slot->set_data(ntt_data);
        }
        NttTaskCoordinates* next_task_c_ptr = ntt_tasks_manager->get_available_task();
        task_slot->set_coordinates(next_task_c_ptr);
        task_slot->dispatch();
      } else {
        // Wait for available tasks
        task_slot = tasks_manager->get_completed_task();
        if (ntt_tasks_manager->handle_completed(task_slot, nof_subntts_l1)) { continue; }
        if (ntt_tasks_manager->available_tasks()) {
          NttTaskCoordinates* next_task_c_ptr = ntt_tasks_manager->get_available_task();
          task_slot->set_coordinates(next_task_c_ptr);
          task_slot->dispatch();
        } else {
          task_slot->set_idle();
        }
      }
    }
    while (true) {
      task_slot = tasks_manager->get_completed_task();
      if (task_slot == nullptr) {
        break;
      } else {
        // ICICLE_LOG_DEBUG << "Task completed: " << task_slot->get_coordinates()->hierarchy_1_layer_idx << ", "
        //                  << task_slot->get_coordinates()->hierarchy_1_subntt_idx << ", "
        //                  << task_slot->get_coordinates()->hierarchy_0_layer_idx << ", "
        //                  << task_slot->get_coordinates()->hierarchy_0_block_idx << ", "
        //                  << task_slot->get_coordinates()->hierarchy_0_subntt_idx;
        task_slot->set_idle();
      }
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Determines if the NTT computation should be parallelized.
   *
   * This function determines if the NTT computation should be parallelized based on the size of the NTT, the batch
   * size, and the size of the scalars.
   *
   * @param logn The log of the size of the NTT.
   * @param config Configuration settings for the NTT, including batch size and scalar size.
   */

  template <typename S, typename E>
  bool NttCpu<S, E>::compute_if_is_parallel(uint32_t logn, const NTTConfig<S>& config)
  {
    // uint32_t log_batch_size = uint32_t(log2(config.batch_size));
    // // for small scalars, the threshold for when it is faster to use parallel NTT is higher
    // if ((scalar_size >= 32 && (logn + log_batch_size) <= 13) || (scalar_size < 32 && (logn + log_batch_size) <= 16)) {
    //   return false;
    // }
    return true;
  }

} // namespace ntt_cpu