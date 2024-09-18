#pragma once
#include "icicle/utils/log.h"
#include "ntt_tasks_manager.h"

using namespace field_config;
using namespace icicle;

namespace ntt_cpu {
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
    */
    template <typename S = scalar_t, typename E = scalar_t>
    class NttCpu
    {
    public:
        NttCpu(uint32_t logn, NTTDir direction, const NTTConfig<S>& config, const E* input, E* output) 
        : input(input),
        ntt_data(logn, output, config, direction),
        // coset_stride(find_or_generate_coset(arbitrary_coset)),
        ntt_tasks_manager(ntt_data.ntt_sub_logn, logn),
        // tasks_manager(std::thread::hardware_concurrency() - 1) // Initialize TasksManager with nof_threads - 1
        tasks_manager(new TasksManager<NttTask<S, E>>(std::thread::hardware_concurrency() - 1)) // TODO - consider using dynamic allocation
        // tasks_manager(std::make_unique<TasksManager<NttTask<S, E>>>(std::thread::hardware_concurrency() - 1)) // Allocate on the heap
        {
          coset_stride = find_or_generate_coset(arbitrary_coset);
        }
        eIcicleError run();
        ~NttCpu() { delete tasks_manager; } // TODO - consider using dynamic allocation


    private:
        const E* input;
        NttData<S, E> ntt_data;
        uint32_t coset_stride;
        std::unique_ptr<S[]> arbitrary_coset = nullptr; // TODO - make const?
        NttTasksManager<S, E> ntt_tasks_manager;
        // TasksManager<NttTask<S, E>> tasks_manager;
        TasksManager<NttTask<S, E>>* tasks_manager; // TODO - consider using dynamic allocation
        // std::unique_ptr<TasksManager<NttTask<S, E>>> tasks_manager; // Change to unique_ptr

        void coset_mul();
        eIcicleError hierarchy1_push_tasks(NttTaskCordinates& ntt_task_cordinates);
        eIcicleError handle_pushed_tasks(uint32_t hierarchy_1_layer_idx);
        void hierarchy_1_reorder();
        eIcicleError reorder_output();
        void reorder_by_bit_reverse();

        E* copy_and_reorder_if_needed(const E* input, E* output);
        uint32_t find_or_generate_coset(std::unique_ptr<S[]>& arbitrary_coset);

    }; // class NttCpu


  //////////////////////////// NttCpu Implementation ////////////////////////////


  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::run()
  {
    NttTaskCordinates ntt_task_cordinates = {0, 0, 0, 0, 0};
    copy_and_reorder_if_needed(input, ntt_data.elements);
    if (ntt_data.config.coset_gen != S::one() && ntt_data.direction == NTTDir::kForward) {
      coset_mul();
    }
    if (ntt_data.ntt_sub_logn.logn > HIERARCHY_1) {
      for (ntt_task_cordinates.hierarchy_1_layer_idx = 0; ntt_task_cordinates.hierarchy_1_layer_idx < 2; ntt_task_cordinates.hierarchy_1_layer_idx++) {
        const uint32_t sunbtt_plus_batch_logn = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx] + uint32_t(log2(ntt_data.config.batch_size));
        const uint32_t log_nof_hierarchy_1_subntts_todo_in_parallel = sunbtt_plus_batch_logn < HIERARCHY_1 ? HIERARCHY_1 - sunbtt_plus_batch_logn : 0;
        const uint32_t nof_hierarchy_1_subntts_todo_in_parallel = 1 << log_nof_hierarchy_1_subntts_todo_in_parallel;
        const uint32_t log_nof_subntts_chunks = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[1 - ntt_task_cordinates.hierarchy_1_layer_idx] -log_nof_hierarchy_1_subntts_todo_in_parallel;
        const uint32_t nof_subntts_chunks = 1 << log_nof_subntts_chunks;
        for (uint32_t hierarchy_1_subntts_chunck_idx = 0; hierarchy_1_subntts_chunck_idx < nof_subntts_chunks; hierarchy_1_subntts_chunck_idx++) {
          for (uint32_t hierarchy_1_subntt_idx_in_chunck = 0; hierarchy_1_subntt_idx_in_chunck < nof_hierarchy_1_subntts_todo_in_parallel; hierarchy_1_subntt_idx_in_chunck++) {
            ntt_task_cordinates.hierarchy_1_subntt_idx = hierarchy_1_subntts_chunck_idx * nof_hierarchy_1_subntts_todo_in_parallel + hierarchy_1_subntt_idx_in_chunck;
            hierarchy1_push_tasks(ntt_task_cordinates);
          }
          handle_pushed_tasks(ntt_task_cordinates.hierarchy_1_layer_idx);
        }
        if (ntt_task_cordinates.hierarchy_1_layer_idx == 0) { hierarchy_1_reorder(); }
      }
      reorder_output();
    } else {
      hierarchy1_push_tasks(ntt_task_cordinates);
      handle_pushed_tasks(0);
    }

    // std::cout << "[NEW] PRE NORMALIZE: left:\t["; for (int i = 0; i < (ntt_data.ntt_sub_logn.size * ntt_data.config.batch_size)-1; i++) { std::cout << ntt_data.elements[i] << ", "; } std::cout <<ntt_data.elements[(ntt_data.ntt_sub_logn.size * ntt_data.config.batch_size)-1]<<"]"<< std::endl;

    if (ntt_data.direction == NTTDir::kInverse) {
      // ICICLE_LOG_DEBUG << "Inverse NTT";
      S inv_size = S::inv_log_size(ntt_data.ntt_sub_logn.logn);
      for (uint64_t i = 0; i < ntt_data.ntt_sub_logn.size * ntt_data.config.batch_size; ++i) {
        ntt_data.elements[i] = ntt_data.elements[i] * inv_size;
      }
      if (ntt_data.config.coset_gen != S::one()) { coset_mul(); }
    }

    if (ntt_data.config.ordering == Ordering::kNR || ntt_data.config.ordering == Ordering::kRR) {
      reorder_by_bit_reverse();
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
   * @note - If `logn > HIERARCHY_1`, there is an additional level of hierarchy due to sub-NTTs, requiring extra
   * reordering
   *       - If `logn <= HIERARCHY_1`, only bit-reversal reordering is applied if configured.
   *       - If no reordering is needed, the input data is directly copied to the output.
   */

  template <typename S, typename E>
  E* NttCpu<S, E>::copy_and_reorder_if_needed(const E* input, E* output)
  {
    const uint64_t total_memory_size = ntt_data.ntt_sub_logn.size * ntt_data.config.batch_size;
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const uint32_t logn = static_cast<uint32_t>(std::log2(ntt_data.ntt_sub_logn.size));
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
      uint32_t cur_ntt_log_size = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
      uint32_t next_ntt_log_size = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[1];

      for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
        const E* input_batch = ntt_data.config.columns_batch ? (input + batch) : (input + batch * ntt_data.ntt_sub_logn.size);
        E* output_batch =
          ntt_data.config.columns_batch ? (temp_output + batch) : (temp_output + batch * ntt_data.ntt_sub_logn.size);

        for (uint64_t i = 0; i < ntt_data.ntt_sub_logn.size; ++i) {
          uint32_t subntt_idx = i >> cur_ntt_log_size;
          uint32_t element = i & ((1 << cur_ntt_log_size) - 1);
          // uint64_t new_idx = bit_rev ? NttUtils<S, E>::bit_reverse(subntt_idx + (element << next_ntt_log_size), logn)
          //                            : subntt_idx + (element << next_ntt_log_size);
          uint64_t new_idx = bit_rev ? bit_reverse(subntt_idx + (element << next_ntt_log_size), logn)
                                     : subntt_idx + (element << next_ntt_log_size);
          output_batch[stride * i] = input_batch[stride * new_idx];
        }
      }

    } else if (bit_rev) {
      // Only bit-reverse reordering needed
      for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
        const E* input_batch = ntt_data.config.columns_batch ? (input + batch) : (input + batch * ntt_data.ntt_sub_logn.size);
        E* output_batch =
          ntt_data.config.columns_batch ? (temp_output + batch) : (temp_output + batch * ntt_data.ntt_sub_logn.size);

        for (uint64_t i = 0; i < ntt_data.ntt_sub_logn.size; ++i) {
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

    return output;
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
  void NttCpu<S, E>::coset_mul()
  {
    uint32_t batch_stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const bool needs_reorder_input = ntt_data.direction == NTTDir::kForward && (ntt_data.ntt_sub_logn.logn > HIERARCHY_1);
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements = ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.ntt_sub_logn.size;

      for (uint64_t i = 1; i < ntt_data.ntt_sub_logn.size; ++i) {
        uint64_t idx = i;

        // Adjust the index if reorder logic was applied on the input
        if (needs_reorder_input) {
          uint32_t cur_ntt_log_size = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
          uint32_t next_ntt_log_size = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
          uint32_t subntt_idx = i >> cur_ntt_log_size;
          uint32_t element = i & ((1 << cur_ntt_log_size) - 1);
          idx = subntt_idx + (element << next_ntt_log_size);
        }

        // Apply coset multiplication based on the available coset information
        if (arbitrary_coset) {
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * arbitrary_coset[idx];
        } else {
          uint32_t twiddle_idx = coset_stride * idx;
          twiddle_idx = ntt_data.direction == NTTDir::kForward ? twiddle_idx : CpuNttDomain<S>::s_ntt_domain.get_max_size() - twiddle_idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * twiddles[twiddle_idx];
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

    if (ntt_data.config.coset_gen != S::one()) {
      try {
        coset_stride =
          CpuNttDomain<S>::s_ntt_domain.get_coset_stride(ntt_data.config.coset_gen); // Coset generator found in twiddles
      } catch (const std::out_of_range& oor) { // Coset generator not found in twiddles. Calculating arbitrary coset
        int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen = 
          ntt_data.direction == NTTDir::kForward ? ntt_data.config.coset_gen : S::inverse(ntt_data.config.coset_gen); // inverse for INTT
        for (uint32_t i = 1; i <= CpuNttDomain<S>::s_ntt_domain.get_max_size(); i++) {
          arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
        }
      }
    }

    return coset_stride;
  }

  /**
   * @brief Reorders elements between layers of hierarchy 1, based on sub-NTT structure.
   */
  template <typename S, typename E>
  void NttCpu<S, E>::hierarchy_1_reorder()
  {
    const uint32_t sntt_size = 1 << ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
    const uint32_t nof_sntts = 1 << ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const uint64_t temp_elements_size = ntt_data.ntt_sub_logn.size * ntt_data.config.batch_size;

    auto temp_elements = std::make_unique<E[]>(temp_elements_size);
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* cur_layer_output = ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.ntt_sub_logn.size;
      E* cur_temp_elements = ntt_data.config.columns_batch ? temp_elements.get() + batch
                                                        : temp_elements.get() + batch * ntt_data.ntt_sub_logn.size;
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
  NttCpu<S, E>::reorder_output()
  {
    uint32_t columns_batch_reps = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    uint32_t rows_batch_reps    = ntt_data.config.columns_batch ? 1 : ntt_data.config.batch_size;
    uint32_t s0 = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
    uint32_t s1 = ntt_data.ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t row_batch = 0; row_batch < rows_batch_reps; ++row_batch) {// if columns_batch=false, then elements pointer is shifted by batch*size
      E* elements = ntt_data.elements + row_batch * ntt_data.ntt_sub_logn.size;
      uint64_t temp_output_size = ntt_data.config.columns_batch ? ntt_data.ntt_sub_logn.size * ntt_data.config.batch_size : ntt_data.ntt_sub_logn.size;
      auto temp_output = std::make_unique<E[]>(temp_output_size);
      uint64_t new_idx = 0;
      uint32_t subntt_idx;
      uint32_t element;
      for (uint32_t col_batch = 0; col_batch < columns_batch_reps; ++col_batch) {
        E* current_elements = ntt_data.config.columns_batch ? elements + col_batch : elements; // if columns_batch=true, then elements pointer is shifted by 1 for each batch
        E* current_temp_output = ntt_data.config.columns_batch ? temp_output.get() + col_batch : temp_output.get();
        for (uint64_t i = 0; i < ntt_data.ntt_sub_logn.size; i++) {
            subntt_idx = i >> s1;
            element = i & ((1 << s1) - 1);
            new_idx = subntt_idx + (element << s0);
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
      std::copy(temp_output.get(), temp_output.get() + temp_output_size, elements); // columns_batch=false: for each row in the batch, copy the reordered elements back to the elements array
    }
    return eIcicleError::SUCCESS;
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
  eIcicleError NttCpu<S, E>::hierarchy1_push_tasks(NttTaskCordinates& ntt_task_cordinates)
  {
    uint32_t nof_hierarchy_0_layers =
      (ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2] != 0)   ? 3
      : (ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1] != 0) ? 2
                                                                                                            : 1;
    uint32_t log_nof_blocks;
    uint32_t log_nof_subntts;
    ntt_task_cordinates.reorder = false;
    for (ntt_task_cordinates.hierarchy_0_layer_idx = 0; ntt_task_cordinates.hierarchy_0_layer_idx < nof_hierarchy_0_layers; ntt_task_cordinates.hierarchy_0_layer_idx++) {
      if (ntt_task_cordinates.hierarchy_0_layer_idx == 0) {
        log_nof_blocks = ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
        log_nof_subntts = ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
      } else if (ntt_task_cordinates.hierarchy_0_layer_idx == 1) {
        log_nof_blocks = ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][2];
        log_nof_subntts = ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0];
      } else {
        log_nof_blocks = ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][0] + ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_cordinates.hierarchy_1_layer_idx][1];
        log_nof_subntts = 0;
        ntt_task_cordinates.hierarchy_0_subntt_idx = 0; // not relevant for layer 2
      }
      for (ntt_task_cordinates.hierarchy_0_block_idx = 0; ntt_task_cordinates.hierarchy_0_block_idx < (1 << log_nof_blocks); ntt_task_cordinates.hierarchy_0_block_idx++) {
        for (ntt_task_cordinates.hierarchy_0_subntt_idx = 0; ntt_task_cordinates.hierarchy_0_subntt_idx < (1 << log_nof_subntts); ntt_task_cordinates.hierarchy_0_subntt_idx++) {
          ntt_tasks_manager.push_task(ntt_task_cordinates);
        }
      }
    }
    if (nof_hierarchy_0_layers > 1) { // all ntt tasks in hierarchy 1 are pushed, now push reorder task so that the data
                                      // is in the correct order for the next hierarchy 1 layer
      ntt_task_cordinates = {ntt_task_cordinates.hierarchy_1_layer_idx, ntt_task_cordinates.hierarchy_1_subntt_idx, nof_hierarchy_0_layers, 0, 0, true};
      ntt_tasks_manager.push_task(ntt_task_cordinates); // reorder=true

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
  eIcicleError NttCpu<S, E>::handle_pushed_tasks(uint32_t hierarchy_1_layer_idx)
  {
    // if (ntt_data.direction == NTTDir::kForward) {
    //   ICICLE_LOG_DEBUG << "handle_pushed_tasks: ntt_data.direction == NTTDir::kForward";
    // }
    // if (ntt_data.direction == NTTDir::kInverse) {
    //   ICICLE_LOG_DEBUG << "handle_pushed_tasks: ntt_data.direction == NTTDir::kInverse";
    // }
    NttTask<S, E>* task_slot = nullptr;

    uint32_t nof_subntts_l1 = 1
                         << ((ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][0]) +
                             (ntt_data.ntt_sub_logn.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1]));
    while (ntt_tasks_manager.tasks_to_do()) {
      // There are tasks that are available or waiting

      if (ntt_tasks_manager.available_tasks()) {
        // Task is available to dispatch
        task_slot = tasks_manager->get_idle_or_completed_task();
        if (task_slot->is_completed()) { ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l1); }
        else {task_slot->set_data(ntt_data);}
        task_slot->set_coordinates(ntt_tasks_manager.get_available_task());
        ntt_tasks_manager.erase_task_from_available_tasks_list();
        task_slot->dispatch();
      } else {
        // Wait for available tasks
        task_slot = tasks_manager->get_completed_task();
        ntt_tasks_manager.set_task_as_completed(*task_slot, nof_subntts_l1);
        if (ntt_tasks_manager.available_tasks()) {
          task_slot->set_coordinates(ntt_tasks_manager.get_available_task());
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

  template <typename S, typename E>
  void NttCpu<S, E>::reorder_by_bit_reverse()
    {
        uint64_t subntt_size = (ntt_data.ntt_sub_logn.size);       
        uint32_t subntt_log_size = (ntt_data.ntt_sub_logn.logn);
        uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
        for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
            E* current_elements = ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.ntt_sub_logn.size;
            uint64_t rev;
            for (uint64_t i = 0; i < subntt_size; ++i) {
                rev = bit_reverse(i, subntt_log_size);
                if (i < rev) {
                    std::swap(current_elements[stride * i], current_elements[stride * rev]);
                    // if (i < ntt_data.ntt_sub_logn.size && rev < ntt_data.ntt_sub_logn.size) { // Ensure indices are within bounds //TODO - Remove after testing
                    //     std::swap(current_elements[stride * i], current_elements[stride * rev]);
                    // } else {
                    //     // Handle out-of-bounds error
                    //     ICICLE_LOG_ERROR << "i=" << i << ", rev=" << rev << ", original_size=" << ntt_data.ntt_sub_logn.size;
                    //     ICICLE_LOG_ERROR << "Index out of bounds: i=" << i << ", rev=" << rev;
                    // }
                }
            }
        }
    }

} // namespace ntt_cpu