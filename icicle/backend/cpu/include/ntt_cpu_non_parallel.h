#pragma once
#include "icicle/utils/log.h"
#include "ntt_tasks_manager.h"
#include "ntt_utils.h"
// #include <_types/_uint32_t.h>
#include <cstdint>
#include <deque>

using namespace field_config;
using namespace icicle;

namespace ntt_cpu {
  /**
   * @brief Constructs an `NttCpuNonParallel` instance with the specified parameters.
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
  class NttCpuNonParallel
  {
  public:
    NttCpuNonParallel(uint32_t logn, NTTDir direction, const NTTConfig<S>& config, const E* input, E* output)
        : input(input), ntt_data(logn, output, config, direction)
    {
    }
    eIcicleError run();

  private:
    const E* input;
    SmallNttData<S, E> ntt_data;

    void coset_mul();
    void reorder_by_bit_reverse();
    void copy_and_reorder_if_needed(const E* input, E* output);
    void ntt8win();
    void ntt16win();
    void ntt32win();
    void hierarchy_0_dit_ntt();

  }; // class NttCpuNonParallel

  //////////////////////////// NttCpuNonParallel Implementation ////////////////////////////

  template <typename S, typename E>
  eIcicleError NttCpuNonParallel<S, E>::run()
  {
    copy_and_reorder_if_needed(input, ntt_data.elements);

    if (ntt_data.direction == NTTDir::kForward && ntt_data.config.coset_gen != S::one()) { coset_mul(); }
    switch (ntt_data.logn) {
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

    if (ntt_data.direction == NTTDir::kInverse && ntt_data.config.coset_gen != S::one()) { coset_mul(); }

    if (ntt_data.config.ordering == Ordering::kNR || ntt_data.config.ordering == Ordering::kRR) {
      reorder_by_bit_reverse();
    }
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Copies the input data to the output buffer with necessary reordering.
   *
   * Depending on the specified ordering configuration,
   * this function performs bit-reversal. If no reordering
   * is needed, the data is copied as-is. It handles scenarios where input and output
   * buffers may overlap by using temporary storage if necessary.
   *
   * @param input  Pointer to the input array to be copied and reordered.
   * @param output Pointer to the output array where the data will be stored.
   * @return E*    Pointer to the output array after copying and reordering.
   */
  template <typename S, typename E>
  void NttCpuNonParallel<S, E>::copy_and_reorder_if_needed(const E* input, E* output)
  {
    const uint64_t total_memory_size = ntt_data.size * ntt_data.config.batch_size;
    const uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const bool bit_rev = ntt_data.config.ordering == Ordering::kRN || ntt_data.config.ordering == Ordering::kRR;

    // Check if input and output point to the same memory location
    E* temp_output = output;
    std::unique_ptr<E[]> temp_storage;
    if (input == output && bit_rev) {
      // Allocate temporary storage to handle in-place reordering
      temp_storage = std::make_unique<E[]>(total_memory_size);
      temp_output = temp_storage.get();
    }

    if (bit_rev) {
      // Only bit-reverse reordering needed
      for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
        const E* input_batch = ntt_data.config.columns_batch ? (input + batch) : (input + batch * ntt_data.size);
        E* output_batch = ntt_data.config.columns_batch ? (temp_output + batch) : (temp_output + batch * ntt_data.size);

        for (uint64_t i = 0; i < ntt_data.size; ++i) {
          uint64_t rev = bit_reverse(i, ntt_data.logn);
          output_batch[stride * i] = input_batch[stride * rev];
        }
      }
    } else {
      // Just copy, no reordering needed
      std::copy(input, input + total_memory_size, output);
    }

    if (input == output && bit_rev) {
      // Copy the reordered elements from the temporary storage back to the output
      std::copy(temp_output, temp_output + total_memory_size, output);
    }
  }

  /**
   * @brief Applies coset multiplication to the elements.
   *
   * This function performs coset multiplication on the `elements` array based on
   * the coset generator specified in the configuration.
   */
  template <typename S, typename E>
  void NttCpuNonParallel<S, E>::coset_mul()
  {
    uint32_t batch_stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;

      for (uint64_t i = 1; i < ntt_data.size; ++i) {
        // Apply coset multiplication based on the available coset information
        if (ntt_data.arbitrary_coset) {
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * ntt_data.arbitrary_coset[i];
        } else {
          uint32_t twiddle_idx = ntt_data.coset_stride * i;
          twiddle_idx = ntt_data.direction == NTTDir::kForward
                          ? twiddle_idx
                          : CpuNttDomain<S>::s_ntt_domain.get_max_size() - twiddle_idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * twiddles[twiddle_idx];
        }
      }
    }
  }

  /**
   * @brief Reorders the elements by performing a bit-reversal permutation.
   *
   * When the configuration specifies bit-reversed ordering (RN or RR), this function
   * rearranges the elements accordingly. It swaps elements whose indices are bitwise
   * reversed with respect to each other, ensuring the output meets the required ordering.
   */
  template <typename S, typename E>
  void NttCpuNonParallel<S, E>::reorder_by_bit_reverse()
  {
    uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * ntt_data.size;
      uint64_t rev;
      for (uint64_t i = 0; i < ntt_data.size; ++i) {
        rev = bit_reverse(i, ntt_data.logn);
        if (i < rev) { std::swap(current_elements[stride * i], current_elements[stride * rev]); }
      }
    }
  }

  /**
   * @brief Performs an optimized NTT transformation using a Winograd approach for size 8.
   *
   * This function applies the Number Theoretic Transform (NTT) on a sub-NTT of size 8 using
   * Winograd algorithm. It utilizes precomputed twiddle factors tailored for
   * the Winograd algorithm to enhance performance. The function handles both forward and inverse
   * transformations based on the NTT direction specified in `ntt_data`.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttCpuNonParallel<S, E>::ntt8win() // N --> N
  {
    const S* twiddles = ntt_data.direction == NTTDir::kForward
                          ? CpuNttDomain<S>::s_ntt_domain.get_winograd8_twiddles()
                          : CpuNttDomain<S>::s_ntt_domain.get_winograd8_twiddles_inv();

    E T;
    uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * (ntt_data.size);

      T = current_elements[stride * 3] - current_elements[stride * 7];
      current_elements[stride * 7] = current_elements[stride * 3] + current_elements[stride * 7];
      current_elements[stride * 3] = current_elements[stride * 1] - current_elements[stride * 5];
      current_elements[stride * 5] = current_elements[stride * 1] + current_elements[stride * 5];
      current_elements[stride * 1] = current_elements[stride * 2] + current_elements[stride * 6];
      current_elements[stride * 2] = current_elements[stride * 2] - current_elements[stride * 6];
      current_elements[stride * 6] = current_elements[stride * 0] + current_elements[stride * 4];
      current_elements[stride * 0] = current_elements[stride * 0] - current_elements[stride * 4];

      current_elements[stride * 2] = current_elements[stride * 2] * twiddles[0];

      current_elements[stride * 4] = current_elements[stride * 6] + current_elements[stride * 1];
      current_elements[stride * 6] = current_elements[stride * 6] - current_elements[stride * 1];
      current_elements[stride * 1] = current_elements[stride * 3] + T;
      current_elements[stride * 3] = current_elements[stride * 3] - T;
      T = current_elements[stride * 5] + current_elements[stride * 7];
      current_elements[stride * 5] = current_elements[stride * 5] - current_elements[stride * 7];
      current_elements[stride * 7] = current_elements[stride * 0] + current_elements[stride * 2];
      current_elements[stride * 0] = current_elements[stride * 0] - current_elements[stride * 2];

      current_elements[stride * 1] = current_elements[stride * 1] * twiddles[1];
      current_elements[stride * 5] = current_elements[stride * 5] * twiddles[0];
      current_elements[stride * 3] = current_elements[stride * 3] * twiddles[2];

      current_elements[stride * 2] = current_elements[stride * 6] + current_elements[stride * 5];
      current_elements[stride * 6] = current_elements[stride * 6] - current_elements[stride * 5];

      current_elements[stride * 5] = current_elements[stride * 1] + current_elements[stride * 3];
      current_elements[stride * 3] = current_elements[stride * 1] - current_elements[stride * 3];

      current_elements[stride * 1] = current_elements[stride * 7] + current_elements[stride * 5];
      current_elements[stride * 5] = current_elements[stride * 7] - current_elements[stride * 5];
      current_elements[stride * 7] = current_elements[stride * 0] - current_elements[stride * 3];
      current_elements[stride * 3] = current_elements[stride * 0] + current_elements[stride * 3];
      current_elements[stride * 0] = current_elements[stride * 4] + T;
      current_elements[stride * 4] = current_elements[stride * 4] - T;

      if (ntt_data.direction == NTTDir::kInverse) {
        S inv_size = S::inv_log_size(ntt_data.logn);
        for (uint64_t i = 0; i < 8; ++i) {
          current_elements[stride * i] = current_elements[stride * i] * inv_size;
        }
      }
    }
  }

  /**
   * @brief Performs an optimized NTT transformation using a Winograd approach for size 16.
   *
   * This function applies the Number Theoretic Transform (NTT) on a sub-NTT of size 16 using
   * Winograd algorithm. It utilizes precomputed twiddle factors tailored for
   * the Winograd algorithm to enhance performance. The function handles both forward and inverse
   * transformations based on the NTT direction.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttCpuNonParallel<S, E>::ntt16win() // N --> N
  {
    const S* twiddles = ntt_data.direction == NTTDir::kForward
                          ? CpuNttDomain<S>::s_ntt_domain.get_winograd16_twiddles()
                          : CpuNttDomain<S>::s_ntt_domain.get_winograd16_twiddles_inv();

    E T;
    uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * (ntt_data.size);

      T = current_elements[stride * 0] + current_elements[stride * 8];
      current_elements[stride * 0] = current_elements[stride * 0] - current_elements[stride * 8];
      current_elements[stride * 8] = current_elements[stride * 4] + current_elements[stride * 12];
      current_elements[stride * 4] = current_elements[stride * 4] - current_elements[stride * 12];
      current_elements[stride * 12] = current_elements[stride * 2] + current_elements[stride * 10];
      current_elements[stride * 2] = current_elements[stride * 2] - current_elements[stride * 10];
      current_elements[stride * 10] = current_elements[stride * 6] + current_elements[stride * 14];
      current_elements[stride * 6] = current_elements[stride * 6] - current_elements[stride * 14];
      current_elements[stride * 14] = current_elements[stride * 1] + current_elements[stride * 9];
      current_elements[stride * 1] = current_elements[stride * 1] - current_elements[stride * 9];
      current_elements[stride * 9] = current_elements[stride * 5] + current_elements[stride * 13];
      current_elements[stride * 5] = current_elements[stride * 5] - current_elements[stride * 13];
      current_elements[stride * 13] = current_elements[stride * 3] + current_elements[stride * 11];
      current_elements[stride * 3] = current_elements[stride * 3] - current_elements[stride * 11];
      current_elements[stride * 11] = current_elements[stride * 7] + current_elements[stride * 15];
      current_elements[stride * 7] = current_elements[stride * 7] - current_elements[stride * 15];
      current_elements[stride * 4] = twiddles[3] * current_elements[stride * 4];

      // 2
      current_elements[stride * 15] = T + current_elements[stride * 8];
      T = T - current_elements[stride * 8];
      current_elements[stride * 8] = current_elements[stride * 0] + current_elements[stride * 4];
      current_elements[stride * 0] = current_elements[stride * 0] - current_elements[stride * 4];
      current_elements[stride * 4] = current_elements[stride * 12] + current_elements[stride * 10];
      current_elements[stride * 12] = current_elements[stride * 12] - current_elements[stride * 10];
      current_elements[stride * 10] = current_elements[stride * 2] + current_elements[stride * 6];
      current_elements[stride * 2] = current_elements[stride * 2] - current_elements[stride * 6];
      current_elements[stride * 6] = current_elements[stride * 14] + current_elements[stride * 9];
      current_elements[stride * 14] = current_elements[stride * 14] - current_elements[stride * 9];
      current_elements[stride * 9] = current_elements[stride * 13] + current_elements[stride * 11];
      current_elements[stride * 13] = current_elements[stride * 13] - current_elements[stride * 11];
      current_elements[stride * 11] = current_elements[stride * 1] + current_elements[stride * 7];
      current_elements[stride * 1] = current_elements[stride * 1] - current_elements[stride * 7];
      current_elements[stride * 7] = current_elements[stride * 3] + current_elements[stride * 5];
      current_elements[stride * 3] = current_elements[stride * 3] - current_elements[stride * 5];

      current_elements[stride * 12] = twiddles[5] * current_elements[stride * 12];
      current_elements[stride * 10] = twiddles[6] * current_elements[stride * 10];
      current_elements[stride * 2] = twiddles[7] * current_elements[stride * 2];

      // 3
      current_elements[stride * 5] = current_elements[stride * 10] + current_elements[stride * 2];
      current_elements[stride * 10] = current_elements[stride * 10] - current_elements[stride * 2];
      current_elements[stride * 2] = current_elements[stride * 6] + current_elements[stride * 9];
      current_elements[stride * 6] = current_elements[stride * 6] - current_elements[stride * 9];
      current_elements[stride * 9] = current_elements[stride * 14] + current_elements[stride * 13];
      current_elements[stride * 14] = current_elements[stride * 14] - current_elements[stride * 13];

      current_elements[stride * 13] = current_elements[stride * 11] + current_elements[stride * 7];
      current_elements[stride * 13] = twiddles[14] * current_elements[stride * 13];
      current_elements[stride * 11] = twiddles[12] * current_elements[stride * 11] + current_elements[stride * 13];
      current_elements[stride * 7] = twiddles[13] * current_elements[stride * 7] + current_elements[stride * 13];

      current_elements[stride * 13] = current_elements[stride * 1] + current_elements[stride * 3];
      current_elements[stride * 13] = twiddles[17] * current_elements[stride * 13];
      current_elements[stride * 1] = twiddles[15] * current_elements[stride * 1] + current_elements[stride * 13];
      current_elements[stride * 3] = twiddles[16] * current_elements[stride * 3] + current_elements[stride * 13];

      // 4
      current_elements[stride * 13] = current_elements[stride * 15] + current_elements[stride * 4];
      current_elements[stride * 15] = current_elements[stride * 15] - current_elements[stride * 4];
      current_elements[stride * 4] = T + current_elements[stride * 12];
      T = T - current_elements[stride * 12];
      current_elements[stride * 12] = current_elements[stride * 8] + current_elements[stride * 5];
      current_elements[stride * 8] = current_elements[stride * 8] - current_elements[stride * 5];
      current_elements[stride * 5] = current_elements[stride * 0] + current_elements[stride * 10];
      current_elements[stride * 0] = current_elements[stride * 0] - current_elements[stride * 10];

      current_elements[stride * 6] = twiddles[9] * current_elements[stride * 6];
      current_elements[stride * 9] = twiddles[10] * current_elements[stride * 9];
      current_elements[stride * 14] = twiddles[11] * current_elements[stride * 14];

      current_elements[stride * 10] = current_elements[stride * 9] + current_elements[stride * 14];
      current_elements[stride * 9] = current_elements[stride * 9] - current_elements[stride * 14];
      current_elements[stride * 14] = current_elements[stride * 11] + current_elements[stride * 1];
      current_elements[stride * 11] = current_elements[stride * 11] - current_elements[stride * 1];
      current_elements[stride * 1] = current_elements[stride * 7] + current_elements[stride * 3];
      current_elements[stride * 7] = current_elements[stride * 7] - current_elements[stride * 3];

      // 5
      current_elements[stride * 3] = current_elements[stride * 13] + current_elements[stride * 2];
      current_elements[stride * 13] = current_elements[stride * 13] - current_elements[stride * 2];
      current_elements[stride * 2] = current_elements[stride * 15] + current_elements[stride * 6];
      current_elements[stride * 15] = current_elements[stride * 15] - current_elements[stride * 6];
      current_elements[stride * 6] = current_elements[stride * 4] + current_elements[stride * 10];
      current_elements[stride * 4] = current_elements[stride * 4] - current_elements[stride * 10];
      current_elements[stride * 10] = T + current_elements[stride * 9];
      T = T - current_elements[stride * 9];
      current_elements[stride * 9] = current_elements[stride * 12] + current_elements[stride * 14];
      current_elements[stride * 12] = current_elements[stride * 12] - current_elements[stride * 14];
      current_elements[stride * 14] = current_elements[stride * 8] + current_elements[stride * 7];
      current_elements[stride * 8] = current_elements[stride * 8] - current_elements[stride * 7];
      current_elements[stride * 7] = current_elements[stride * 5] + current_elements[stride * 1];
      current_elements[stride * 5] = current_elements[stride * 5] - current_elements[stride * 1];
      current_elements[stride * 1] = current_elements[stride * 0] + current_elements[stride * 11];
      current_elements[stride * 0] = current_elements[stride * 0] - current_elements[stride * 11];

      // reorder + return
      current_elements[stride * 11] = current_elements[stride * 0];
      current_elements[stride * 0] = current_elements[stride * 3];
      current_elements[stride * 3] = current_elements[stride * 7];
      current_elements[stride * 7] = current_elements[stride * 1];
      current_elements[stride * 1] = current_elements[stride * 9];
      current_elements[stride * 9] = current_elements[stride * 12];
      current_elements[stride * 12] = current_elements[stride * 15];
      current_elements[stride * 15] = current_elements[stride * 11];
      current_elements[stride * 11] = current_elements[stride * 5];
      current_elements[stride * 5] = current_elements[stride * 14];
      current_elements[stride * 14] = T;
      T = current_elements[stride * 8];
      current_elements[stride * 8] = current_elements[stride * 13];
      current_elements[stride * 13] = T;
      T = current_elements[stride * 4];
      current_elements[stride * 4] = current_elements[stride * 2];
      current_elements[stride * 2] = current_elements[stride * 6];
      current_elements[stride * 6] = current_elements[stride * 10];
      current_elements[stride * 10] = T;

      if (ntt_data.direction == NTTDir::kInverse) {
        S inv_size = S::inv_log_size(ntt_data.logn);
        for (uint64_t i = 0; i < 16; ++i) {
          current_elements[stride * i] = current_elements[stride * i] * inv_size;
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
   * transformations based on the NTT direction.
   *
   * @return void
   */
  template <typename S, typename E>
  void NttCpuNonParallel<S, E>::ntt32win() // N --> N
  {
    const S* twiddles = ntt_data.direction == NTTDir::kForward
                          ? CpuNttDomain<S>::s_ntt_domain.get_winograd32_twiddles()
                          : CpuNttDomain<S>::s_ntt_domain.get_winograd32_twiddles_inv();

    std::vector<E> temp_0(46);
    std::vector<E> temp_1(46);
    uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * (ntt_data.size);
      /*  Stage s00  */
      temp_0[0] = current_elements[stride * 0];
      temp_0[1] = current_elements[stride * 2];
      temp_0[2] = current_elements[stride * 4];
      temp_0[3] = current_elements[stride * 6];
      temp_0[4] = current_elements[stride * 8];
      temp_0[5] = current_elements[stride * 10];
      temp_0[6] = current_elements[stride * 12];
      temp_0[7] = current_elements[stride * 14];
      temp_0[8] = current_elements[stride * 16];
      temp_0[9] = current_elements[stride * 18];
      temp_0[10] = current_elements[stride * 20];
      temp_0[11] = current_elements[stride * 22];
      temp_0[12] = current_elements[stride * 24];
      temp_0[13] = current_elements[stride * 26];
      temp_0[14] = current_elements[stride * 28];
      temp_0[15] = current_elements[stride * 30];
      temp_0[16] = current_elements[stride * 1];
      temp_0[17] = current_elements[stride * 3];
      temp_0[18] = current_elements[stride * 5];
      temp_0[19] = current_elements[stride * 7];
      temp_0[20] = current_elements[stride * 9];
      temp_0[21] = current_elements[stride * 11];
      temp_0[22] = current_elements[stride * 13];
      temp_0[23] = current_elements[stride * 15];
      temp_0[24] = current_elements[stride * 17];
      temp_0[25] = current_elements[stride * 19];
      temp_0[26] = current_elements[stride * 21];
      temp_0[27] = current_elements[stride * 23];
      temp_0[28] = current_elements[stride * 25];
      temp_0[29] = current_elements[stride * 27];
      temp_0[30] = current_elements[stride * 29];
      temp_0[31] = current_elements[stride * 31];

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

      current_elements[stride * 0] = temp_1[0] + temp_1[16];
      current_elements[stride * 1] = temp_1[1] + temp_1[17];
      current_elements[stride * 2] = temp_1[18] + temp_1[2];
      current_elements[stride * 3] = temp_1[19] + temp_1[3];
      current_elements[stride * 4] = temp_1[20] + temp_1[4];
      current_elements[stride * 5] = temp_1[21] + temp_1[5];
      current_elements[stride * 6] = temp_1[22] + temp_1[6];
      current_elements[stride * 7] = temp_1[23] + temp_1[7];
      current_elements[stride * 8] = temp_1[24] + temp_1[8];
      current_elements[stride * 9] = temp_1[25] + temp_1[9];
      current_elements[stride * 10] = temp_1[10] + temp_1[26];
      current_elements[stride * 11] = temp_1[11] + temp_1[27];
      current_elements[stride * 12] = temp_1[12] + temp_1[28];
      current_elements[stride * 13] = temp_1[13] + temp_1[29];
      current_elements[stride * 14] = temp_1[14] + temp_1[30];
      current_elements[stride * 15] = temp_1[15] + temp_1[31];
      current_elements[stride * 16] = temp_1[0] - temp_1[16];
      current_elements[stride * 17] = temp_1[1] - temp_1[17];
      current_elements[stride * 18] = temp_1[2] - temp_1[18];
      current_elements[stride * 19] = temp_1[3] - temp_1[19];
      current_elements[stride * 20] = temp_1[4] - temp_1[20];
      current_elements[stride * 21] = temp_1[5] - temp_1[21];
      current_elements[stride * 22] = temp_1[6] - temp_1[22];
      current_elements[stride * 23] = temp_1[7] - temp_1[23];
      current_elements[stride * 24] = temp_1[8] - temp_1[24];
      current_elements[stride * 25] = temp_1[9] - temp_1[25];
      current_elements[stride * 26] = temp_1[10] - temp_1[26];
      current_elements[stride * 27] = temp_1[11] - temp_1[27];
      current_elements[stride * 28] = temp_1[12] - temp_1[28];
      current_elements[stride * 29] = temp_1[13] - temp_1[29];
      current_elements[stride * 30] = temp_1[14] - temp_1[30];
      current_elements[stride * 31] = temp_1[15] - temp_1[31];

      if (ntt_data.direction == NTTDir::kInverse) {
        S inv_size = S::inv_log_size(ntt_data.logn);
        for (uint64_t i = 0; i < 32; ++i) {
          current_elements[stride * i] = current_elements[stride * i] * inv_size;
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
  void NttCpuNonParallel<S, E>::hierarchy_0_dit_ntt() // R --> N
  {
    uint32_t offset = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();

    uint32_t stride = ntt_data.config.columns_batch ? ntt_data.config.batch_size : 1;
    for (uint32_t batch = 0; batch < ntt_data.config.batch_size; ++batch) {
      E* current_elements =
        ntt_data.config.columns_batch ? ntt_data.elements + batch : ntt_data.elements + batch * (ntt_data.size);
      for (uint32_t len = 2; len <= ntt_data.size; len <<= 1) {
        uint32_t half_len = len / 2;
        uint32_t step = (ntt_data.size / len) * (CpuNttDomain<S>::s_ntt_domain.get_max_size() >> ntt_data.logn);
        for (uint32_t i = 0; i < ntt_data.size; i += len) {
          for (uint32_t j = 0; j < half_len; ++j) {
            uint64_t u_mem_idx = stride * (i + j);
            uint64_t v_mem_idx = stride * (i + j + half_len);
            E u = current_elements[u_mem_idx];
            E v;
            if (j == 0) {
              v = current_elements[v_mem_idx];
            } else {
              uint32_t tw_idx = (ntt_data.direction == NTTDir::kForward)
                                  ? j * step
                                  : CpuNttDomain<S>::s_ntt_domain.get_max_size() - j * step;
              v = current_elements[v_mem_idx] * twiddles[tw_idx];
            }
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = u - v;
          }
        }
      }

      if (ntt_data.direction == NTTDir::kInverse) {
        S inv_size = S::inv_log_size(ntt_data.logn);
        for (uint64_t i = 0; i < ntt_data.size; ++i) {
          current_elements[stride * i] = current_elements[stride * i] * inv_size;
        }
      }
    }
  }

} // namespace ntt_cpu