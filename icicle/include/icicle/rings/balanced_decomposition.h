#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {

  /**
   * @brief Decomposes elements in T into balanced base-b digits.
   *
   * For each input element x ∈ T, this function computes a sequence of digits
   * r_i ∈ (-b/2, b/2] (interpreted in T) such that:
   *
   *     x ≡ ∑ r_i * b^i mod q
   *
   * The number of digits per input element is implicitly computed as ceil(log_b(q)).
   *
   * @tparam T Element type of the ring or field (e.g., uint64_t, int64_t).
   *
   * @param input         Pointer to input elements.
   * @param input_size    Number of elements to decompose.
                          If `config.batch_size > 1`, this should be a concatenated array of vectors.
   * @param base          The base b for decomposition (must fit in uint32_t).
   * @param config        Configuration for the operation.
   * @param output        Output buffer to store balanced digits per input element.
   *                      Must have space for input_size * num_digits(base, q) elements.
   *                      If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *
   * @return              eIcicleError indicating success or failure.
   */
  template <typename T>
  eIcicleError
  decompose_balanced_digits(const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output);

  /**
   * @brief Recomposes canonical T elements from balanced base-b digits.
   *
   * Given a sequence of digits r_i ∈ (-b/2, b/2] per element,
   * this function reconstructs:
   *
   *     x ≡ ∑ r_i * b^i mod q
   *
   * and returns the result in canonical T representation.
   *
   * @tparam T Element type of the ring or field (e.g., uint64_t, int64_t).
   *
   * @param input         Pointer to input digits (flattened array).
   *                      Must contain output_size * num_digits(base, q) elements.
   *                      If `config.batch_size > 1`, this should be a concatenated array of vectors.
   * @param output_size   Number of elements to recompose.
   * @param base          The base b used in decomposition.
   * @param config        Configuration for the operation.
   * @param output        Output buffer to store recomposed elements.
   *                      Must have space for output_size elements.
   *                      If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *
   * @return              eIcicleError indicating success or failure.
   */
  template <typename T>
  eIcicleError recompose_from_balanced_digits(
    const T* input, size_t output_size, uint32_t base, const VecOpsConfig& config, T* output);

} // namespace icicle