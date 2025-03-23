#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {

  /**
   * @brief Decomposes elements in Zq into balanced base-b digits.
   *
   * For each input element x ∈ Zq,
   * this function computes a sequence of digits r_i ∈ (-b/2, b/2] (∈ Zq)
   * such that:
   *
   *    x ≡ ∑ r_i * b^i mod q
   *
   * The number of digits per input element is implicitly computed as ceil(log_b(q))
   *
   * @tparam Zq Integer-ring
   *
   * @param input         Pointer to input elements.
   * @param input_size    Number of field elements to decompose.
   * @param base          The base b for decomposition (must fit in uint32_t).
   * @param config        Configuration for the operation.
   * @param output        Output buffer: stores balanced digits for each input.
   *                      Must have space for input_size * num_digits(base, q) elements.
   *
   * @return              eIcicleError indicating success or failure.
   */
  template <typename Zq>
  eIcicleError
  decompose_balanced_digits(const Zq* input, size_t input_size, uint32_t base, const VecopsConfig& config, Zq* output);

  /**
   * @brief Recomposes elements from balanced base-b digits into canonical Zq elements.
   *
   * Given a sequence of digits r_i ∈ (-b/2, b/2] per element,
   * this function reconstructs:
   *
   *    x ≡ ∑ r_i * b^i mod q
   *
   * and stores the result in canonical Zq representation.
   *
   * @tparam Zq Integer-ring
   *
   * @param input         Pointer to input digits (flattened array).
   *                      Expecting 'output_size * num_digits(base, q)' elements
   * @param output_size   Number of output elements to recompose.
   * @param base          The base b used in decomposition.
   * @param config        Configuration for the operation.
   * @param output        Output buffer for recomposed field elements (as limbs).
   *                      Must have space for output_size.
   *
   * @return              eIcicleError indicating success or failure.
   */
  template <typename Zq>
  eIcicleError recompose_from_balanced_digits(
    const Zq* input, size_t output_size, uint32_t base, const VecopsConfig& config, Zq* output);
} // namespace icicle