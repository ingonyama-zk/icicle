#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {

  namespace balanced_decomposition {
    /**
     * @brief Compute the number of digits required to represent any element in Z_q
     *        using balanced base decomposition with the given base.
     *
     * Each field element (in [0, q)) may be represented using digits in the range
     * [-b/2, b/2), where b = base. This method returns the number of such digits
     * needed to fully represent any field element.
     *
     * If base > 2, we add one extra digit to safely account for any additional
     * carry caused by shifting digits into the balanced range (e.g., digit > b/2).
     *
     * @tparam T    Field element type (must match field_t::TLC == 2).
     * @param base  The base to use for balanced decomposition (must be >= 2).
     * @return      Number of digits needed for full representation.
     */
    template <typename T>
    static constexpr inline uint32_t compute_nof_digits(uint32_t base)
    {
      static_assert(T::TLC == 2, "Balanced decomposition assumes q ~64-bit");

      // Get the modulus q as an int64_t
      constexpr auto q_storage = T::get_modulus();
      const int64_t q = *(const int64_t*)&q_storage;
      ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

      // Compute minimum number of digits based on log(q) / log(base)
      const double log2_q = std::log2(static_cast<double>(q));
      const double log2_base = std::log2(static_cast<double>(base));
      const uint32_t base_digits = static_cast<uint32_t>(std::ceil(log2_q / log2_base));

      // For base > 2, we may need an extra digit due to the carry when balancing
      return base > 2 ? base_digits + 1 : base_digits;
    }

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
 * @param output_size    Number of output digits.
 *
 * @return              eIcicleError indicating success or failure.
 */
    template <typename T>
    eIcicleError decompose(
      const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size);

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
     * @param input_size    Number of input digits.
     * @param base          The base b used in decomposition.
     * @param config        Configuration for the operation.
     * @param output        Output buffer to store recomposed elements.
     *                      Must have space for output_size elements.
     *                      If `config.batch_size > 1`, this should be a concatenated array of vectors.
     * @param output_size   Number of elements to recompose.
     *
     * @return              eIcicleError indicating success or failure.
     */
    template <typename T>
    eIcicleError recompose(
      const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size);
  } // namespace balanced_decomposition

} // namespace icicle
