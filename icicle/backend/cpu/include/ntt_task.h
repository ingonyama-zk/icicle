#pragma once
#include "icicle/ntt.h"
#include "icicle/utils/log.h"
#include "ntt_utils.h"
#include "ntt_data.h"
#include <cstdint>
#include <random>

using namespace field_config;
using namespace icicle;
namespace ntt_cpu {

  /**
   * @brief Represents a task in the NTT computation, handling either NTT calculation or reordering.
   *
   * This class manages tasks within the NTT computation, performing either the NTT computation
   * for a given sub-NTT or reordering the output if required.
   *
   * @method void execute() Executes the task, either performing the NTT computation or reordering the output.
   * @method NttTaskCoordinates get_coordinates() const Returns the task's coordinates.
   * @method void set_coordinates(NttTaskParams<S, E> params) Sets the task parameters.
   * @method void set_data(NttData<S, E>& data) Sets the NTT data for the task.
   */
  template <typename S = scalar_t, typename E = scalar_t>
  class NttTask : public TaskBase
  {
  public:
    NttTask() : ntt_data(nullptr) {}

    void execute();
    NttTaskCoordinates* get_coordinates() const
    {
      return ntt_task_coordinates;
    } // Returns a pointer to the `NttTaskCoordinates` structure that specifies the task's position within the NTT
      // computation hierarchy.
    void set_coordinates(NttTaskCoordinates* task_c_ptr)
    {
      ntt_task_coordinates = task_c_ptr;
    } // Assigns a pointer to a `NttTaskCoordinates` structure, which specifies the task's position within the NTT
      // computation hierarchy.
    void set_data(NttData<S, E>& data)
    {
      ntt_data = &data;
    } // Sets the `NttData` structure that contains all necessary data and configurations required.

  private:
    NttTaskCoordinates* ntt_task_coordinates = nullptr;
    NttData<S, E>* ntt_data = nullptr;
    eIcicleError reorder_and_refactor_if_needed();
    void apply_coset_multiplication(E* current_elements, const std::vector<uint32_t>& index_in_mem, const S* twiddles);
    eIcicleError hierarchy_0_cpu_ntt();
    void ntt8win();
    void ntt16win();
    void ntt32win();
    void hierarchy_0_dit_ntt();
    void hierarchy_0_dif_ntt();
    void reorder_by_bit_reverse();
    void refactor_output_hierarchy_0();
    uint64_t idx_in_mem(NttTaskCoordinates* ntt_task_coordinates, uint32_t element);
  };

  //////////////////////////// NttTask Implementation ////////////////////////////

  /**
   * @brief Executes the NTT task.
   *
   * This function determines the type of task to perform based on the `reorder` flag in `ntt_task_coordinates`.
   * - If `reorder` is `false` (most tasks), it performs the NTT computation by invoking `hierarchy_0_cpu_ntt()`.
   * - If `reorder` is `true`, it handles reordering and refactoring by calling `reorder_and_refactor_if_needed()`.
   * @return void
   */
  template <typename S, typename E>
  void NttTask<S, E>::execute()
  {
    if (!ntt_task_coordinates->reorder) {
      hierarchy_0_cpu_ntt();
    } else {
      // if all hierarchy_0_subntts are done, and at least 2 layers in hierarchy 0 - reorder the subntt's output
      reorder_and_refactor_if_needed();
    }
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
   * @return eIcicleError::SUCCESS on successful execution.
   */
  template <typename S, typename E>
  eIcicleError NttTask<S, E>::reorder_and_refactor_if_needed()
  {
    uint32_t columns_batch_reps = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    uint32_t rows_batch_reps = ntt_data->config.columns_batch ? 1 : ntt_data->config.batch_size;
    for (uint32_t row_batch = 0; row_batch < rows_batch_reps;
         ++row_batch) { // if columns_batch=false, then elements pointer is shifted by batch*size
      E* elements = ntt_data->elements + row_batch * ntt_data->ntt_sub_logn.size;
      bool is_only_hierarchy_0 = ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[0] == 0;
      const bool refactor_pre_hierarchy_1_next_layer =
        (!is_only_hierarchy_0) && (ntt_task_coordinates->hierarchy_1_layer_idx == 0);
      uint64_t size =
        (is_only_hierarchy_0)
          ? ntt_data->ntt_sub_logn.size
          : 1 << ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx];
      uint64_t temp_output_size = ntt_data->config.columns_batch ? size * ntt_data->config.batch_size : size;
      auto temp_output = std::make_unique<E[]>(temp_output_size);
      uint64_t new_idx = 0;
      uint32_t subntt_idx;
      uint32_t element;
      uint32_t s0 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][0];
      uint32_t s1 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][1];
      uint32_t s2 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][2];
      uint32_t p0, p1, p2;
      const uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
      uint32_t rep = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
      uint64_t tw_idx = 0;
      const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
      E* hierarchy_1_subntt_output =
        elements +
        stride * (ntt_task_coordinates->hierarchy_1_subntt_idx
                  << ntt_data->ntt_sub_logn
                       .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx
                                                                                                   // * subntt_size
      for (uint32_t col_batch = 0; col_batch < columns_batch_reps; ++col_batch) {
        E* current_elements =
          ntt_data->config.columns_batch
            ? hierarchy_1_subntt_output + col_batch
            : hierarchy_1_subntt_output; // if columns_batch=true, then elements pointer is shifted by 1 for each batch
        E* current_temp_output = ntt_data->config.columns_batch ? temp_output.get() + col_batch : temp_output.get();
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
            tw_idx = (ntt_data->direction == NTTDir::kForward)
                       ? ((CpuNttDomain<S>::s_ntt_domain.get_max_size() >> ntt_data->ntt_sub_logn.logn) *
                          ntt_task_coordinates->hierarchy_1_subntt_idx * new_idx)
                       : CpuNttDomain<S>::s_ntt_domain.get_max_size() -
                           ((CpuNttDomain<S>::s_ntt_domain.get_max_size() >> ntt_data->ntt_sub_logn.logn) *
                            ntt_task_coordinates->hierarchy_1_subntt_idx * new_idx);
            current_temp_output[stride * new_idx] = current_elements[stride * i] * twiddles[tw_idx];
          } else {
            current_temp_output[stride * new_idx] = current_elements[stride * i];
          }
        }
      }
      std::copy(
        temp_output.get(), temp_output.get() + temp_output_size,
        hierarchy_1_subntt_output); // columns_batch=false: for each row in the batch, copy the reordered elements back
                                    // to the elements array
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Applies coset multiplication to the current elements of the NTT computation.
   *
   * This function multiplies the current elements with the appropriate coset factors based on
   * their indices. It handles both predefined and arbitrary coset multiplications depending
   * on the availability of coset stride information.
   *
   * @param current_elements Pointer to the array of current elements being processed.
   * @param index_in_mem Vector containing the memory indices of the elements to be multiplied.
   * @param twiddles Pointer to the array of twiddle factors used for multiplication.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttTask<S, E>::apply_coset_multiplication(
    E* current_elements, const std::vector<uint32_t>& index_in_mem, const S* twiddles)
  {
    uint32_t current_subntt_size =
      1 << ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                             [ntt_task_coordinates->hierarchy_0_layer_idx];
    uint32_t subntt_idx;
    uint32_t s0 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][0];
    uint32_t s1 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][1];
    uint32_t s2 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][2];
    uint32_t p0, p1, p2;
    for (uint32_t i = 0; i < current_subntt_size; i++) {
      uint64_t new_idx = i;
      uint64_t idx = idx_in_mem(ntt_task_coordinates, i); // don't need to multiply by stride here
      // Adjust the index if reorder logic was applied on the input
      if (ntt_data->ntt_sub_logn.logn > HIERARCHY_1) {
        uint32_t cur_ntt_log_size = ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[0];
        uint32_t next_ntt_log_size = ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[1];
        uint32_t subntt_idx = index_in_mem[i] >> cur_ntt_log_size;
        uint32_t element = index_in_mem[i] & ((1 << cur_ntt_log_size) - 1);
        idx = subntt_idx + (element << next_ntt_log_size);
      }
      // Apply coset multiplication based on the available coset information
      if (ntt_data->arbitrary_coset) {
        current_elements[index_in_mem[new_idx]] =
          current_elements[index_in_mem[new_idx]] * ntt_data->arbitrary_coset[idx];
      } else {
        uint32_t twiddle_idx = ntt_data->coset_stride * idx;
        twiddle_idx = ntt_data->direction == NTTDir::kForward
                        ? twiddle_idx
                        : CpuNttDomain<S>::s_ntt_domain.get_max_size() - twiddle_idx;
        current_elements[index_in_mem[new_idx]] = current_elements[index_in_mem[new_idx]] * twiddles[twiddle_idx];
      }
    }
  }

  /**
   * @brief Executes the NTT on a sub-NTT at the hierarchy_0 level.
   *
   * This function applies the NTT on a sub-NTT specified by the task coordinates at the hierarchy_0 level.
   *  Depending on the sub-NTT size, it may utilize optimized Winograd NTT functions
   * (`ntt8win`, `ntt16win`, `ntt32win`) or fall back to a general DIT NTT implementation.
   *
   * If further refactoring is required after the NTT operation, it prepares the output for the next
   * hierarchy layer by invoking `refactor_output_hierarchy_0()`.
   *
   * @return eIcicleError::SUCCESS on successful execution.
   */
  template <typename S, typename E>
  eIcicleError NttTask<S, E>::hierarchy_0_cpu_ntt()
  {
    const uint64_t subntt_size =
      (1 << ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                              [ntt_task_coordinates->hierarchy_0_layer_idx]);
    uint64_t original_size = (ntt_data->ntt_sub_logn.size);
    const uint64_t total_memory_size = original_size * ntt_data->config.batch_size;
    const uint32_t subntt_size_log =
      ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                        [ntt_task_coordinates->hierarchy_0_layer_idx];
    switch (subntt_size_log) {
    case 3:
      ntt8win();
      break;
    case 4:
      ntt16win();
      break;
    case 5:
      ntt32win();
      break;
    default:
      reorder_by_bit_reverse();
      hierarchy_0_dit_ntt(); // R --> N
      break;
    }

    if (
      ntt_task_coordinates->hierarchy_0_layer_idx != 2 &&
      ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                        [ntt_task_coordinates->hierarchy_0_layer_idx + 1] != 0) {
      refactor_output_hierarchy_0();
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Performs an optimized NTT transformation using a Winograd approach for size 8.
   *
   * This function applies the Number Theoretic Transform (NTT) on a sub-NTT of size 8 using
   * a specialized Winograd algorithm. It utilizes precomputed twiddle factors tailored for
   * the Winograd algorithm to enhance performance. The function handles both forward and inverse
   * transformations based on the NTT direction specified in `ntt_data`.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttTask<S, E>::ntt8win() // N --> N
  {
    uint32_t offset = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    E* subntt_elements =
      ntt_data->elements +
      offset * (ntt_task_coordinates->hierarchy_1_subntt_idx
                << ntt_data->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
    const S* twiddles = ntt_data->direction == NTTDir::kForward
                          ? CpuNttDomain<S>::s_ntt_domain.get_winograd8_twiddles()
                          : CpuNttDomain<S>::s_ntt_domain.get_winograd8_twiddles_inv();

    E T;
    std::vector<uint32_t> index_in_mem(8);
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    for (uint32_t i = 0; i < 8; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_coordinates, i);
    }
    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* current_elements = ntt_data->config.columns_batch ? subntt_elements + batch
                                                           : subntt_elements + batch * (ntt_data->ntt_sub_logn.size);

      if (
        ntt_task_coordinates->hierarchy_1_layer_idx == 0 && ntt_task_coordinates->hierarchy_0_layer_idx == 0 &&
        ntt_data->config.coset_gen != S::one() && ntt_data->direction == NTTDir::kForward) {
        apply_coset_multiplication(current_elements, index_in_mem, CpuNttDomain<S>::s_ntt_domain.get_twiddles());
      }

      T = current_elements[index_in_mem[3]] - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[3]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[1]] - current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[2]] + current_elements[index_in_mem[6]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[2]] - current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[4]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[4]];

      current_elements[index_in_mem[2]] = current_elements[index_in_mem[2]] * twiddles[0];

      current_elements[index_in_mem[4]] = current_elements[index_in_mem[6]] + current_elements[index_in_mem[1]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[6]] - current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[3]] + T;
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[3]] - T;
      T = current_elements[index_in_mem[5]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[5]] - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[2]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[2]];

      current_elements[index_in_mem[1]] = current_elements[index_in_mem[1]] * twiddles[1];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[5]] * twiddles[0];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[3]] * twiddles[2];

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

      bool last_layer =
        (ntt_task_coordinates->hierarchy_1_layer_idx == 1 ||
         (ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[1] == 0)) &&
        (ntt_task_coordinates->hierarchy_0_layer_idx == 2 ||
         (ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                            [ntt_task_coordinates->hierarchy_0_layer_idx + 1] == 0));
      if (last_layer && ntt_data->direction == NTTDir::kInverse) {
        const S* inv_log_sizes = CpuNttDomain<S>::s_ntt_domain.get_inv_log_sizes();
        S inv_size = inv_log_sizes[ntt_data->ntt_sub_logn.logn];
        for (uint64_t i = 0; i < 8; ++i) {
          current_elements[index_in_mem[i]] = current_elements[index_in_mem[i]] * inv_size;
        }
      }
    }
  }

  /**
   * @brief Performs an optimized NTT transformation using a Winograd approach for size 16.
   *
   * This function applies the Number Theoretic Transform (NTT) on a sub-NTT of size 16 using
   * a specialized Winograd algorithm. It utilizes precomputed twiddle factors tailored for
   * the Winograd algorithm to enhance performance. The function handles both forward and inverse
   * transformations based on the NTT direction specified in `ntt_data`.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttTask<S, E>::ntt16win() // N --> N
  {
    uint32_t offset = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    E* subntt_elements =
      ntt_data->elements +
      offset * (ntt_task_coordinates->hierarchy_1_subntt_idx
                << ntt_data->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
    const S* twiddles = ntt_data->direction == NTTDir::kForward
                          ? CpuNttDomain<S>::s_ntt_domain.get_winograd16_twiddles()
                          : CpuNttDomain<S>::s_ntt_domain.get_winograd16_twiddles_inv();

    E T;
    std::vector<uint32_t> index_in_mem(16);
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    for (uint32_t i = 0; i < 16; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_coordinates, i);
    }
    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* current_elements = ntt_data->config.columns_batch ? subntt_elements + batch
                                                           : subntt_elements + batch * (ntt_data->ntt_sub_logn.size);

      if (
        ntt_task_coordinates->hierarchy_1_layer_idx == 0 && ntt_task_coordinates->hierarchy_0_layer_idx == 0 &&
        ntt_data->config.coset_gen != S::one() && ntt_data->direction == NTTDir::kForward) {
        apply_coset_multiplication(current_elements, index_in_mem, CpuNttDomain<S>::s_ntt_domain.get_twiddles());
      }

      T = current_elements[index_in_mem[0]] + current_elements[index_in_mem[8]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[8]];
      current_elements[index_in_mem[8]] = current_elements[index_in_mem[4]] + current_elements[index_in_mem[12]];
      current_elements[index_in_mem[4]] = current_elements[index_in_mem[4]] - current_elements[index_in_mem[12]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[2]] + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[2]] - current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = current_elements[index_in_mem[6]] + current_elements[index_in_mem[14]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[6]] - current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[9]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[1]] - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]] = current_elements[index_in_mem[5]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[5]] - current_elements[index_in_mem[13]];
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[3]] + current_elements[index_in_mem[11]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[3]] - current_elements[index_in_mem[11]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[7]] + current_elements[index_in_mem[15]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[7]] - current_elements[index_in_mem[15]];
      current_elements[index_in_mem[4]] = twiddles[3] * current_elements[index_in_mem[4]];

      // 2
      current_elements[index_in_mem[15]] = T + current_elements[index_in_mem[8]];
      T = T - current_elements[index_in_mem[8]];
      current_elements[index_in_mem[8]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[4]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[4]];
      current_elements[index_in_mem[4]] = current_elements[index_in_mem[12]] + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[12]] - current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = current_elements[index_in_mem[2]] + current_elements[index_in_mem[6]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[2]] - current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[14]] + current_elements[index_in_mem[9]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[14]] - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]] = current_elements[index_in_mem[13]] + current_elements[index_in_mem[11]];
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[13]] - current_elements[index_in_mem[11]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[1]] - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[3]] + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[3]] - current_elements[index_in_mem[5]];

      current_elements[index_in_mem[12]] = twiddles[5] * current_elements[index_in_mem[12]];
      current_elements[index_in_mem[10]] = twiddles[6] * current_elements[index_in_mem[10]];
      current_elements[index_in_mem[2]] = twiddles[7] * current_elements[index_in_mem[2]];

      // 3
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[10]] + current_elements[index_in_mem[2]];
      current_elements[index_in_mem[10]] = current_elements[index_in_mem[10]] - current_elements[index_in_mem[2]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[6]] + current_elements[index_in_mem[9]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[6]] - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]] = current_elements[index_in_mem[14]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[14]] - current_elements[index_in_mem[13]];

      current_elements[index_in_mem[13]] = current_elements[index_in_mem[11]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[13]] = twiddles[14] * current_elements[index_in_mem[13]];
      current_elements[index_in_mem[11]] =
        twiddles[12] * current_elements[index_in_mem[11]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[7]] =
        twiddles[13] * current_elements[index_in_mem[7]] + current_elements[index_in_mem[13]];

      current_elements[index_in_mem[13]] = current_elements[index_in_mem[1]] + current_elements[index_in_mem[3]];
      current_elements[index_in_mem[13]] = twiddles[17] * current_elements[index_in_mem[13]];
      current_elements[index_in_mem[1]] =
        twiddles[15] * current_elements[index_in_mem[1]] + current_elements[index_in_mem[13]];
      current_elements[index_in_mem[3]] =
        twiddles[16] * current_elements[index_in_mem[3]] + current_elements[index_in_mem[13]];

      // 4
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[15]] + current_elements[index_in_mem[4]];
      current_elements[index_in_mem[15]] = current_elements[index_in_mem[15]] - current_elements[index_in_mem[4]];
      current_elements[index_in_mem[4]] = T + current_elements[index_in_mem[12]];
      T = T - current_elements[index_in_mem[12]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[8]] + current_elements[index_in_mem[5]];
      current_elements[index_in_mem[8]] = current_elements[index_in_mem[8]] - current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[10]];

      current_elements[index_in_mem[6]] = twiddles[9] * current_elements[index_in_mem[6]];
      current_elements[index_in_mem[9]] = twiddles[10] * current_elements[index_in_mem[9]];
      current_elements[index_in_mem[14]] = twiddles[11] * current_elements[index_in_mem[14]];

      current_elements[index_in_mem[10]] = current_elements[index_in_mem[9]] + current_elements[index_in_mem[14]];
      current_elements[index_in_mem[9]] = current_elements[index_in_mem[9]] - current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[11]] + current_elements[index_in_mem[1]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[11]] - current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[7]] + current_elements[index_in_mem[3]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[7]] - current_elements[index_in_mem[3]];

      // 5
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[13]] + current_elements[index_in_mem[2]];
      current_elements[index_in_mem[13]] = current_elements[index_in_mem[13]] - current_elements[index_in_mem[2]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[15]] + current_elements[index_in_mem[6]];
      current_elements[index_in_mem[15]] = current_elements[index_in_mem[15]] - current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[4]] + current_elements[index_in_mem[10]];
      current_elements[index_in_mem[4]] = current_elements[index_in_mem[4]] - current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = T + current_elements[index_in_mem[9]];
      T = T - current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]] = current_elements[index_in_mem[12]] + current_elements[index_in_mem[14]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[12]] - current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = current_elements[index_in_mem[8]] + current_elements[index_in_mem[7]];
      current_elements[index_in_mem[8]] = current_elements[index_in_mem[8]] - current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[5]] + current_elements[index_in_mem[1]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[5]] - current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[0]] + current_elements[index_in_mem[11]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[0]] - current_elements[index_in_mem[11]];

      // reorder + return
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[0]];
      current_elements[index_in_mem[0]] = current_elements[index_in_mem[3]];
      current_elements[index_in_mem[3]] = current_elements[index_in_mem[7]];
      current_elements[index_in_mem[7]] = current_elements[index_in_mem[1]];
      current_elements[index_in_mem[1]] = current_elements[index_in_mem[9]];
      current_elements[index_in_mem[9]] = current_elements[index_in_mem[12]];
      current_elements[index_in_mem[12]] = current_elements[index_in_mem[15]];
      current_elements[index_in_mem[15]] = current_elements[index_in_mem[11]];
      current_elements[index_in_mem[11]] = current_elements[index_in_mem[5]];
      current_elements[index_in_mem[5]] = current_elements[index_in_mem[14]];
      current_elements[index_in_mem[14]] = T;
      T = current_elements[index_in_mem[8]];
      current_elements[index_in_mem[8]] = current_elements[index_in_mem[13]];
      current_elements[index_in_mem[13]] = T;
      T = current_elements[index_in_mem[4]];
      current_elements[index_in_mem[4]] = current_elements[index_in_mem[2]];
      current_elements[index_in_mem[2]] = current_elements[index_in_mem[6]];
      current_elements[index_in_mem[6]] = current_elements[index_in_mem[10]];
      current_elements[index_in_mem[10]] = T;

      bool last_layer =
        (ntt_task_coordinates->hierarchy_1_layer_idx == 1 ||
         (ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[1] == 0)) &&
        (ntt_task_coordinates->hierarchy_0_layer_idx == 2 ||
         (ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                            [ntt_task_coordinates->hierarchy_0_layer_idx + 1] == 0));
      if (last_layer && ntt_data->direction == NTTDir::kInverse) {
        const S* inv_log_sizes = CpuNttDomain<S>::s_ntt_domain.get_inv_log_sizes();
        S inv_size = inv_log_sizes[ntt_data->ntt_sub_logn.logn];
        for (uint64_t i = 0; i < 16; ++i) {
          current_elements[index_in_mem[i]] = current_elements[index_in_mem[i]] * inv_size;
        }
      }
    }
  }

  /**
   * @brief Performs an optimized NTT transformation using a Winograd approach for size 32.
   *
   * This function applies the Number Theoretic Transform (NTT) on a sub-NTT of size 32 using
   * a specialized Winograd algorithm. It utilizes precomputed twiddle factors tailored for
   * the Winograd algorithm to enhance performance. The function handles both forward and inverse
   * transformations based on the NTT direction specified in `ntt_data`.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttTask<S, E>::ntt32win() // N --> N
  {
    uint32_t offset = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    E* subntt_elements =
      ntt_data->elements +
      offset * (ntt_task_coordinates->hierarchy_1_subntt_idx
                << ntt_data->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
    const S* twiddles = ntt_data->direction == NTTDir::kForward
                          ? CpuNttDomain<S>::s_ntt_domain.get_winograd32_twiddles()
                          : CpuNttDomain<S>::s_ntt_domain.get_winograd32_twiddles_inv();

    std::vector<E> temp_0(46);
    std::vector<E> temp_1(46);
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    std::vector<uint32_t> index_in_mem(32);
    for (uint32_t i = 0; i < 32; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_coordinates, i);
    }

    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* current_elements = ntt_data->config.columns_batch ? subntt_elements + batch
                                                           : subntt_elements + batch * (ntt_data->ntt_sub_logn.size);

      if (
        ntt_task_coordinates->hierarchy_1_layer_idx == 0 && ntt_task_coordinates->hierarchy_0_layer_idx == 0 &&
        ntt_data->config.coset_gen != S::one() && ntt_data->direction == NTTDir::kForward) {
        apply_coset_multiplication(current_elements, index_in_mem, CpuNttDomain<S>::s_ntt_domain.get_twiddles());
      }

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

      // multiply by winograd twiddles, skip if twiddle is 1
      for (uint32_t i = 0; i < 3; i++) {
        temp_0[i] = temp_1[i];
      }
      temp_0[3] = temp_1[3] * twiddles[3];
      temp_0[4] = temp_1[4];
      for (uint32_t i = 5; i < 8; i++) {
        temp_0[i] = temp_1[i] * twiddles[i];
      }
      temp_0[8] = temp_1[8];
      for (uint32_t i = 9; i < 18; i++) {
        temp_0[i] = temp_1[i] * twiddles[i];
      }
      temp_0[18] = temp_1[18];
      for (uint32_t i = 19; i < 46; i++) {
        temp_0[i] = temp_1[i] * twiddles[i];
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

      bool last_layer =
        (ntt_task_coordinates->hierarchy_1_layer_idx == 1 ||
         (ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[1] == 0)) &&
        (ntt_task_coordinates->hierarchy_0_layer_idx == 2 ||
         (ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                            [ntt_task_coordinates->hierarchy_0_layer_idx + 1] == 0));
      if (last_layer && ntt_data->direction == NTTDir::kInverse) {
        const S* inv_log_sizes = CpuNttDomain<S>::s_ntt_domain.get_inv_log_sizes();
        S inv_size = inv_log_sizes[ntt_data->ntt_sub_logn.logn];
        for (uint64_t i = 0; i < 32; ++i) {
          current_elements[index_in_mem[i]] = current_elements[index_in_mem[i]] * inv_size;
        }
      }
    }
  }

  /**
   * @brief Performs the Decimation-In-Time (DIT) NTT transform on a sub-NTT.
   *
   * This function applies the Decimation-In-Time (DIT) Number Theoretic Transform (NTT) to
   * the specified sub-NTT, transforming the data from the bit-reversed order (R) to natural order (N).
   * The transformation is performed iteratively by dividing the sub-NTT into smaller segments, applying
   * butterfly operations, and utilizing twiddle factors.
   *
   */
  template <typename S, typename E>
  void NttTask<S, E>::hierarchy_0_dit_ntt() // R --> N
  {
    uint32_t offset = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    E* subntt_elements =
      ntt_data->elements +
      offset * (ntt_task_coordinates->hierarchy_1_subntt_idx
                << ntt_data->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    const uint32_t subntt_size_log =
      ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                        [ntt_task_coordinates->hierarchy_0_layer_idx];
    const uint64_t subntt_size = 1 << subntt_size_log;
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    std::vector<uint32_t> index_in_mem(subntt_size);
    for (uint32_t i = 0; i < subntt_size; i++) {
      index_in_mem[i] = stride * idx_in_mem(ntt_task_coordinates, i);
    }
    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* current_elements = ntt_data->config.columns_batch ? subntt_elements + batch
                                                           : subntt_elements + batch * (ntt_data->ntt_sub_logn.size);
      if (
        ntt_task_coordinates->hierarchy_1_layer_idx == 0 && ntt_task_coordinates->hierarchy_0_layer_idx == 0 &&
        ntt_data->config.coset_gen != S::one() && ntt_data->direction == NTTDir::kForward) {
        apply_coset_multiplication(current_elements, index_in_mem, twiddles);
      }

      for (uint32_t len = 2; len <= subntt_size; len <<= 1) {
        uint32_t half_len = len / 2;
        uint32_t step = (subntt_size / len) * (CpuNttDomain<S>::s_ntt_domain.get_max_size() >> subntt_size_log);
        for (uint32_t i = 0; i < subntt_size; i += len) {
          for (uint32_t j = 0; j < half_len; ++j) {
            uint64_t u_mem_idx = index_in_mem[i + j];
            uint64_t v_mem_idx = index_in_mem[i + j + half_len];
            E u = current_elements[u_mem_idx];
            E v;
            if (j == 0) {
              v = current_elements[v_mem_idx];
            } else {
              uint32_t tw_idx = (ntt_data->direction == NTTDir::kForward)
                                  ? j * step
                                  : CpuNttDomain<S>::s_ntt_domain.get_max_size() - j * step;
              v = current_elements[v_mem_idx] * twiddles[tw_idx];
            }
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = u - v;
          }
        }
      }

      bool last_layer =
        (ntt_task_coordinates->hierarchy_1_layer_idx == 1 ||
         (ntt_data->ntt_sub_logn.hierarchy_1_layers_sub_logn[1] == 0)) &&
        (ntt_task_coordinates->hierarchy_0_layer_idx == 2 ||
         (ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                            [ntt_task_coordinates->hierarchy_0_layer_idx + 1] == 0));
      if (last_layer && ntt_data->direction == NTTDir::kInverse) {
        uint32_t current_subntt_size =
          1 << ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                                 [ntt_task_coordinates->hierarchy_0_layer_idx];
        S inv_size = S::inv_log_size(ntt_data->ntt_sub_logn.logn);
        for (uint64_t i = 0; i < current_subntt_size; ++i) {
          current_elements[index_in_mem[i]] = current_elements[index_in_mem[i]] * inv_size;
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
   */
  template <typename S, typename E>
  void NttTask<S, E>::hierarchy_0_dif_ntt() // N --> R
  {
    uint32_t offset = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    E* subntt_elements =
      ntt_data->elements +
      offset * (ntt_task_coordinates->hierarchy_1_subntt_idx
                << ntt_data->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    const uint32_t subntt_size_log =
      ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                        [ntt_task_coordinates->hierarchy_0_layer_idx];
    const uint64_t subntt_size = 1 << subntt_size_log;
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* current_elements = ntt_data->config.columns_batch ? subntt_elements + batch
                                                           : subntt_elements + batch * (ntt_data->ntt_sub_logn.size);
      for (uint32_t len = subntt_size; len >= 2; len >>= 1) {
        uint32_t half_len = len / 2;
        uint32_t step = (subntt_size / len) * (CpuNttDomain<S>::s_ntt_domain.get_max_size() >> subntt_size_log);
        for (uint32_t i = 0; i < subntt_size; i += len) {
          for (uint32_t j = 0; j < half_len; ++j) {
            uint64_t u_mem_idx = stride * idx_in_mem(ntt_task_coordinates, i + j);
            uint64_t v_mem_idx = stride * idx_in_mem(ntt_task_coordinates, i + j + half_len);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx];
            current_elements[u_mem_idx] = u + v;
            if (j == 0) {
              current_elements[v_mem_idx] = (u - v);
            } else {
              uint32_t tw_idx = (ntt_data->direction == NTTDir::kForward)
                                  ? j * step
                                  : CpuNttDomain<S>::s_ntt_domain.get_max_size() - j * step;
              current_elements[v_mem_idx] = (u - v) * twiddles[tw_idx];
            }
          }
        }
      }
    }
  }

  /**
   * @brief Reorders elements by bit-reversing their indices within a sub-NTT.
   *
   * This function reorders the elements of a sub-NTT based on the bit-reversed indices.
   * The reordering is performed either on the entire NTT or within a specific sub-NTT,
   * depending on whether the operation is at the top hierarchy level. The function accesses
   * corrected memory addresses because reordering between layers of hierarchy 0 was skipped.
   *
   */
  template <typename S, typename E>
  void NttTask<S, E>::reorder_by_bit_reverse()
  {
    uint32_t offset = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    E* subntt_elements =
      ntt_data->elements +
      offset * (ntt_task_coordinates->hierarchy_1_subntt_idx
                << ntt_data->ntt_sub_logn
                     .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx *
                                                                                                 // subntt_size
    uint64_t subntt_size =
      1 << ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                             [ntt_task_coordinates->hierarchy_0_layer_idx];
    uint32_t subntt_log_size =
      ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                        [ntt_task_coordinates->hierarchy_0_layer_idx];
    uint64_t original_size = (1 << ntt_data->ntt_sub_logn.logn);
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* current_elements = ntt_data->config.columns_batch ? subntt_elements + batch
                                                           : subntt_elements + batch * ntt_data->ntt_sub_logn.size;
      uint64_t rev;
      uint64_t i_mem_idx;
      uint64_t rev_mem_idx;
      for (uint64_t i = 0; i < subntt_size; ++i) {
        // rev = NttUtils<S, E>::bit_reverse(i, subntt_log_size);
        rev = bit_reverse(i, subntt_log_size);
        i_mem_idx = idx_in_mem(ntt_task_coordinates, i);
        rev_mem_idx = idx_in_mem(ntt_task_coordinates, rev);
        if (i < rev) {
          if (i_mem_idx < ntt_data->ntt_sub_logn.size && rev_mem_idx < ntt_data->ntt_sub_logn.size) { // Ensure indices
                                                                                                      // are
                                                                                                      // within bounds
            std::swap(current_elements[stride * i_mem_idx], current_elements[stride * rev_mem_idx]);
          } else {
            // Handle out-of-bounds error
            ICICLE_LOG_ERROR << "i=" << i << ", rev=" << rev << ", original_size=" << ntt_data->ntt_sub_logn.size;
            ICICLE_LOG_ERROR << "Index out of bounds: i_mem_idx=" << i_mem_idx << ", rev_mem_idx=" << rev_mem_idx;
          }
        }
      }
    }
  }

  /**
   * @brief Refactors the output of an hierarchy_0 sub-NTT after the NTT operation.
   *
   * This function refactors the output of an hierarchy_0 sub-NTT by applying twiddle factors to the elements
   * based on their indices. It prepares the data for further processing in subsequent layers of the NTT hierarchy.
   * Accesses corrected memory addresses, because reordering between layers of hierarchy 0 was skipped.
   */
  template <typename S, typename E>
  void NttTask<S, E>::refactor_output_hierarchy_0()
  {
    uint32_t hierarchy_0_subntt_size =
      1 << ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]
                                                             [ntt_task_coordinates->hierarchy_0_layer_idx];
    uint32_t hierarchy_0_nof_subntts =
      1 << ntt_data->ntt_sub_logn
             .hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][0]; // only relevant for layer 1
    uint32_t i, j, i_0;
    uint32_t ntt_size =
      ntt_task_coordinates->hierarchy_0_layer_idx == 0
        ? 1
            << (ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][0] +
                ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][1])
        : 1
            << (ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][0] +
                ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][1] +
                ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][2]);
    uint32_t stride = ntt_data->config.columns_batch ? ntt_data->config.batch_size : 1;
    uint64_t original_size = (1 << ntt_data->ntt_sub_logn.logn);
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    for (uint32_t batch = 0; batch < ntt_data->config.batch_size; ++batch) {
      E* hierarchy_1_subntt_elements =
        ntt_data->elements +
        stride * (ntt_task_coordinates->hierarchy_1_subntt_idx
                  << ntt_data->ntt_sub_logn
                       .hierarchy_1_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx]); // input + subntt_idx
                                                                                                   // * subntt_size
      E* elements_of_current_batch = ntt_data->config.columns_batch
                                       ? hierarchy_1_subntt_elements + batch
                                       : hierarchy_1_subntt_elements + batch * original_size;
      for (uint32_t elem = 0; elem < hierarchy_0_subntt_size; elem++) {
        uint64_t elem_mem_idx = stride * idx_in_mem(ntt_task_coordinates, elem);
        i = (ntt_task_coordinates->hierarchy_0_layer_idx == 0)
              ? elem
              : elem * hierarchy_0_nof_subntts + ntt_task_coordinates->hierarchy_0_subntt_idx;
        j = (ntt_task_coordinates->hierarchy_0_layer_idx == 0) ? ntt_task_coordinates->hierarchy_0_subntt_idx
                                                               : ntt_task_coordinates->hierarchy_0_block_idx;
        uint64_t tw_idx = (ntt_data->direction == NTTDir::kForward)
                            ? ((CpuNttDomain<S>::s_ntt_domain.get_max_size() / ntt_size) * j * i)
                            : CpuNttDomain<S>::s_ntt_domain.get_max_size() -
                                ((CpuNttDomain<S>::s_ntt_domain.get_max_size() / ntt_size) * j * i);
        elements_of_current_batch[elem_mem_idx] = elements_of_current_batch[elem_mem_idx] * twiddles[tw_idx];
      }
    }
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
   * @param ntt_task_coordinates The coordinates specifying the current task within the NTT hierarchy.
   * @param element_idx The specific element index within the sub-NTT.
   * @return uint64_t The computed memory index for the given element.
   */

  template <typename S, typename E>
  uint64_t NttTask<S, E>::idx_in_mem(NttTaskCoordinates* ntt_task_coordinates, uint32_t element_idx)
  {
    uint32_t s0 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][0];
    uint32_t s1 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][1];
    uint32_t s2 = ntt_data->ntt_sub_logn.hierarchy_0_layers_sub_logn[ntt_task_coordinates->hierarchy_1_layer_idx][2];
    switch (ntt_task_coordinates->hierarchy_0_layer_idx) {
    case 0:
      return ntt_task_coordinates->hierarchy_0_block_idx +
             ((ntt_task_coordinates->hierarchy_0_subntt_idx + (element_idx << s1)) << s2);
    case 1:
      return ntt_task_coordinates->hierarchy_0_block_idx +
             ((element_idx + (ntt_task_coordinates->hierarchy_0_subntt_idx << s1)) << s2);
    case 2:
      return ((ntt_task_coordinates->hierarchy_0_block_idx << (s1 + s2)) & ((1 << (s0 + s1 + s2)) - 1)) +
             (((ntt_task_coordinates->hierarchy_0_block_idx << (s1 + s2)) >> (s0 + s1 + s2)) << s2) + element_idx;
    default:
      ICICLE_ASSERT(false) << "Unsupported layer";
    }
    return static_cast<uint64_t>(-1);
  }

} // namespace ntt_cpu