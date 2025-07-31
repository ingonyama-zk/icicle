#pragma once
#include "icicle/errors.h"
#include "ntt_task.h"
#include "ntt_utils.h"
#include <taskflow/taskflow.hpp>
#include <cstdint>

#ifdef CURVE_ID
  #include "icicle/curves/curve_config.h"
using namespace curve_config;
  #define IS_ECNTT std::is_same_v<E, projective_t>
#else
  #define IS_ECNTT false
#endif

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
    E* output;
    std::unique_ptr<E[]> temp_elements;
    NttData<S, E> ntt_data;

    bool compute_if_is_parallel(uint32_t logn, const NTTConfig<S>& config);
    void coset_mul();
    void reorder_by_bit_reverse();
    void copy_and_reorder_if_needed(const E* input, E* output);
    void hierarchy_1_reorder();
    eIcicleError reorder_output();

  }; // class NttCpu

  //////////////////////////// NttCpu Implementation ////////////////////////////

  /*
   * @brief Constructs an `NttCpu` instance
   */
  template <typename S, typename E>
  NttCpu<S, E>::NttCpu(uint32_t logn, NTTDir direction, const NTTConfig<S>& config, const E* input, E* output)
      : input(input), output(output), ntt_data(logn, output, config, direction, compute_if_is_parallel(logn, config))
  {
    if (logn > HIERARCHY_1) {
      // Allocate temporary storage to handle reordering
      temp_elements = std::make_unique<E[]>(ntt_data.size * config.batch_size);
    }
  }

  template <typename S, typename E>
  eIcicleError NttCpu<S, E>::run()
  {
    copy_and_reorder_if_needed(input, ntt_data.elements);
    if (ntt_data.direction == NTTDir::kForward && ntt_data.config.coset_gen != S::one()) { coset_mul(); }
    if (!ntt_data.is_parallel) {
      NttTaskCoordinates ntt_task_coordinates(0, 0, 0, 0, 0, false);
      NttTask<S, E> task(ntt_task_coordinates, ntt_data);
      task.execute();
    } else if (__builtin_expect((ntt_data.logn <= HIERARCHY_1), 1)) {
      tf::Taskflow taskflow;
      tf::Executor executor;
      uint32_t nof_hierarchy_0_layers = (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][2] != 0)   ? 3
                                        : (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][1] != 0) ? 2
                                                                                                                : 1;
      for (uint32_t hierarchy_0_layer_idx = 0; hierarchy_0_layer_idx < nof_hierarchy_0_layers;
           hierarchy_0_layer_idx++) {
        uint64_t nof_blocks;
        uint64_t nof_subntts;
        if (hierarchy_0_layer_idx == 0) {
          nof_blocks = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][2];
          nof_subntts = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][1];
        } else if (hierarchy_0_layer_idx == 1) {
          nof_blocks = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][2];
          nof_subntts = 1 << ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][0];
        } else {
          nof_blocks = 1
                       << (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][0] +
                           ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[0][1]);
          nof_subntts = 1;
        }
        size_t num_chunks = (std::thread::hardware_concurrency()) << 1; // Adjust based on the number of threads
        size_t chunk_size = (nof_blocks * nof_subntts + num_chunks - 1) / num_chunks;

        for (size_t i = 0; i < num_chunks; ++i) {
          size_t start_index = i * chunk_size;
          size_t end_index = std::min(start_index + chunk_size, static_cast<size_t>(nof_blocks * nof_subntts));
          taskflow.emplace([&, hierarchy_0_layer_idx, start_index, end_index, nof_subntts]() {
            for (uint32_t j = start_index; j < (end_index); j++) {
              uint32_t hierarchy_0_block_idx = j / nof_subntts;
              uint32_t hierarchy_0_subntt_idx = j % nof_subntts;
              NttTaskCoordinates ntt_task_coordinates(
                0, 0, hierarchy_0_layer_idx, hierarchy_0_block_idx, hierarchy_0_subntt_idx, false);
              NttTask<S, E> task(ntt_task_coordinates, ntt_data);
              task.execute();
            }
          });
        }
        executor.run(taskflow).wait(); // TODO: Explore using task dependencies to optimize parallel execution
        taskflow.clear();
        if ((hierarchy_0_layer_idx != 0) && (hierarchy_0_layer_idx == nof_hierarchy_0_layers - 1)) { // All NTT tasks in
                                                                                                     // hierarchy 1 have
                                                                                                     // been executed;
                                                                                                     // now executing
                                                                                                     // the reorder task
          NttTaskCoordinates ntt_task_coordinates(0, 0, hierarchy_0_layer_idx, 0, 0, true);
          NttTask<S, E> task(ntt_task_coordinates, ntt_data);
          task.execute();
        }
      }
    } else {
      tf::Taskflow taskflow;
      tf::Executor executor;
      for (uint32_t hierarchy_1_layer_idx = 0; hierarchy_1_layer_idx < 2; hierarchy_1_layer_idx++) {
        const uint32_t sunbtt_plus_batch_logn =
          ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[hierarchy_1_layer_idx] +
          uint32_t(log2(ntt_data.config.batch_size));
        const uint32_t log_nof_hierarchy_1_subntts_todo_in_parallel =
          (sunbtt_plus_batch_logn < HIERARCHY_1) ? HIERARCHY_1 - sunbtt_plus_batch_logn : 0;
        const uint32_t nof_hierarchy_1_subntts_todo_in_parallel = 1 << log_nof_hierarchy_1_subntts_todo_in_parallel;
        const uint32_t log_nof_subntts_chunks =
          ntt_data.ntt_sub_hierarchies.hierarchy_1_layers_sub_logn[1 - hierarchy_1_layer_idx] -
          log_nof_hierarchy_1_subntts_todo_in_parallel;
        const uint32_t nof_subntts_chunks = 1 << log_nof_subntts_chunks;

        for (uint32_t hierarchy_1_subntts_chunk_idx = 0; hierarchy_1_subntts_chunk_idx < nof_subntts_chunks;
             hierarchy_1_subntts_chunk_idx++) {
          uint32_t nof_hierarchy_0_layers =
            (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][2] != 0)   ? 3
            : (ntt_data.ntt_sub_hierarchies.hierarchy_0_layers_sub_logn[hierarchy_1_layer_idx][1] != 0) ? 2
                                                                                                        : 1;
          for (uint32_t hierarchy_0_layer_idx = 0; hierarchy_0_layer_idx < nof_hierarchy_0_layers;
               hierarchy_0_layer_idx++) {
            uint64_t nof_blocks;
            uint64_t nof_subntts;
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
            nof_blocks = nof_blocks >> 1;
            size_t num_chunks = (std::thread::hardware_concurrency()) << 5; // Adjust based on the number of threads
            size_t chunk_size =
              (nof_blocks * nof_subntts * nof_hierarchy_1_subntts_todo_in_parallel + num_chunks - 1) / num_chunks;
            for (size_t i = 0; i < num_chunks; ++i) {
              size_t start_index = i * chunk_size;
              size_t end_index = std::min(
                start_index + chunk_size,
                static_cast<size_t>(nof_blocks * nof_subntts * nof_hierarchy_1_subntts_todo_in_parallel));
              taskflow.emplace([this, start_index, end_index, nof_blocks, hierarchy_1_layer_idx,
                                hierarchy_1_subntts_chunk_idx, hierarchy_0_layer_idx, nof_subntts,
                                nof_hierarchy_1_subntts_todo_in_parallel]() {
                for (uint32_t j = start_index; j < (end_index); j++) {
                  uint32_t hierarchy_1_subntt_idx_in_chunk = j / (nof_subntts * nof_blocks);
                  uint32_t hierarchy_0_block_idx_half = (j / nof_subntts) % nof_blocks;
                  uint32_t hierarchy_0_block_idx = hierarchy_0_block_idx_half << 1;
                  uint32_t hierarchy_0_subntt_idx = j % nof_subntts;
                  NttTaskCoordinates ntt_task_coordinates(
                    hierarchy_1_layer_idx,
                    hierarchy_1_subntts_chunk_idx * nof_hierarchy_1_subntts_todo_in_parallel +
                      hierarchy_1_subntt_idx_in_chunk,
                    hierarchy_0_layer_idx, hierarchy_0_block_idx, hierarchy_0_subntt_idx, false);
                  NttTask<S, E> task(ntt_task_coordinates, ntt_data);
                  task.execute();
                  ntt_task_coordinates.hierarchy_0_block_idx = hierarchy_0_block_idx + 1;
                  NttTask<S, E> task_with_elements_in_the_same_cachline(ntt_task_coordinates, ntt_data);
                  task_with_elements_in_the_same_cachline.execute();
                }
              });
            }
            executor.run(taskflow).wait();
            taskflow.clear();
            if (
              (hierarchy_0_layer_idx != 0) &&
              (hierarchy_0_layer_idx ==
               nof_hierarchy_0_layers -
                 1)) { // All NTT tasks in hierarchy 1 have been executed; now executing the reorder task
              taskflow.emplace([this, hierarchy_1_layer_idx, hierarchy_1_subntts_chunk_idx,
                                nof_hierarchy_1_subntts_todo_in_parallel, nof_hierarchy_0_layers]() {
                for (uint32_t hierarchy_1_subntt_idx_in_chunk = 0;
                     hierarchy_1_subntt_idx_in_chunk < nof_hierarchy_1_subntts_todo_in_parallel;
                     hierarchy_1_subntt_idx_in_chunk++) {
                  NttTaskCoordinates ntt_task_coordinates(
                    hierarchy_1_layer_idx,
                    hierarchy_1_subntts_chunk_idx * nof_hierarchy_1_subntts_todo_in_parallel +
                      hierarchy_1_subntt_idx_in_chunk,
                    nof_hierarchy_0_layers, 0, 0, true);
                  NttTask<S, E> task(ntt_task_coordinates, ntt_data);
                  task.execute();
                }
              });
              executor.run(taskflow).wait();
              taskflow.clear();
            }
          }
        }
        if (hierarchy_1_layer_idx == 0) { hierarchy_1_reorder(); }
      }
      reorder_output();
    }
    if (ntt_data.direction == NTTDir::kInverse && ntt_data.config.coset_gen != S::one()) { coset_mul(); }

    if (
      ntt_data.config.ordering == Ordering::kNR || ntt_data.config.ordering == Ordering::kNM ||
      ntt_data.config.ordering == Ordering::kRR) {
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
    const bool bit_rev = ntt_data.config.ordering == Ordering::kRN || ntt_data.config.ordering == Ordering::kRR ||
                         ntt_data.config.ordering == Ordering::kMN;

    // Check if input and output point to the same memory location
    E* temp_output = output;
    std::unique_ptr<E[]> temp_storage;
    if (input == output) {
      if (!(logn > HIERARCHY_1 || bit_rev)) {
        // no reordering needed, and input and output are the same
        return;
      }
      // Allocate temporary storage to handle in-place reordering, can't be done inplace when input and output are the
      // same
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
    tf::Executor executor;
    tf::Taskflow taskflow;

    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;

      size_t num_chunks = (std::thread::hardware_concurrency()) << 1; // Adjust based on the number of threads
      size_t chunk_size = (ntt_data.size + num_chunks - 1) / num_chunks;

      for (size_t cunk = 0; cunk < num_chunks; ++cunk) {
        size_t start_index = cunk * chunk_size;
        size_t end_index = std::min(start_index + chunk_size, static_cast<size_t>(ntt_data.size));
        taskflow.emplace([&, needs_reorder_input, twiddles, current_elements, batch_stride, start_index, end_index]() {
          for (uint64_t i = start_index; i < end_index; i++) {
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
        });
      }
      executor.run(taskflow).wait();
      taskflow.clear();
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
    tf::Executor executor;
    tf::Taskflow taskflow;

    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* cur_layer_output =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;
      E* cur_temp_elements =
        ntt_data.config.columns_batch ? temp_elements.get() + batch : temp_elements.get() + batch * ntt_data.size;
      size_t num_chunks = std::thread::hardware_concurrency(); // Adjust based on the number of threads
      size_t chunk_size = (static_cast<size_t>(nof_sntts * sntt_size) + num_chunks - 1) / num_chunks;
      for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start_index = chunk * chunk_size;
        size_t end_index = std::min(start_index + chunk_size, static_cast<size_t>(nof_sntts * sntt_size));
        taskflow.emplace(
          [this, stride, cur_layer_output, cur_temp_elements, nof_sntts, sntt_size, start_index, end_index]() {
            for (size_t j = start_index; j < end_index; ++j) {
              uint32_t sntt_idx = j / sntt_size;
              uint32_t elem = j % sntt_size;
              cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
                cur_layer_output[stride * (elem * nof_sntts + sntt_idx)];
            }
          });
      }
      executor.run(taskflow).wait();
      taskflow.clear();
    }
    ntt_data.elements = temp_elements.get();
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
      uint64_t new_idx = 0;
      uint32_t subntt_idx;
      uint32_t element;
      for (uint32_t col_batch = 0; col_batch < columns_batch_reps; ++col_batch) {
        E* current_elements =
          ntt_data.config.columns_batch
            ? elements + col_batch
            : elements; // if columns_batch=true, then elements pointer is shifted by 1 for each batch
        E* current_temp_output = ntt_data.config.columns_batch ? output + col_batch : output;
        for (uint64_t i = 0; i < ntt_data.size; i++) {
          subntt_idx = i >> s1;
          element = i & ((1 << s1) - 1);
          new_idx = subntt_idx + (element << s0);
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
    }
    ntt_data.elements = output;
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
    uint32_t log_batch_size = uint32_t(log2(config.batch_size));
    // For ecntt we want parallelism unless really small case
    if constexpr (IS_ECNTT) {
      return logn > 5;
    } else {
      uint32_t scalar_size = sizeof(S);
      // for small scalars, the threshold for when it is faster to use parallel NTT is higher
      if ((scalar_size >= 32 && (logn + log_batch_size) <= 11) || (scalar_size < 32 && (logn + log_batch_size) <= 16)) {
        return false;
      }
    }
    return true;
  }

} // namespace ntt_cpu