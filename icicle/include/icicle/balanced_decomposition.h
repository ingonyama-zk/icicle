#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {
  namespace balanced_decomposition {

    /**
     * @brief Balanced base-b decomposition for elements in Zq or polynomial rings over Zq.
     *
     * Supported types:
     *   - Scalar rings: Zq
     *   - Polynomial rings: Rq = Zq[x]/(X^d + 1), modeled as PolynomialRing<Zq, d>
     *
     * In all cases, the layout is **digit-major**:
     *   - The output is organized as a matrix where each row corresponds to a digit index.
     *   - Each row contains all elements’ values at that digit position (e.g., r₀, then r₁, etc.).
     */

    /**
     * @brief Compute the number of digits required to represent any Zq element using balanced base-b decomposition.
     *
     * The result applies per scalar coefficient and extends naturally to polynomials by applying it to each
     * coefficient.
     *
     * Each Zq element is represented with digits in the range (-b/2, b/2], and an extra digit is added when `base > 2`
     * to account for possible carry propagation.
     *
     * @tparam T    Element type.
     * @param base  Decomposition base b (must be ≥ 2).
     * @return      Number of digits required to represent a single Zq element in balanced base-b form.
     */
    template <typename T>
    static constexpr inline uint32_t compute_nof_digits(uint32_t base)
    {
      static_assert(T::TLC == 2, "Balanced decomposition assumes q ~64-bit");

      constexpr auto q_storage = T::get_modulus();
      const int64_t q = *(const int64_t*)&q_storage;
      ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

      const double log2_q = std::log2(static_cast<double>(q));
      const double log2_b = std::log2(static_cast<double>(base));
      const uint32_t digits = static_cast<uint32_t>(std::ceil(log2_q / log2_b));

      return base > 2 ? digits + 1 : digits;
    }

    /**
     * @brief Decompose a vector of elements into balanced base-b digits.
     *
     * For Zq scalars:
     *   Each element x ∈ Zq is decomposed as:
     *
     *     x ≡ r₀ + b·r₁ + b²·r₂ + ... + b^{t−1}·r_{t−1} mod q
     *
     *   The output is digit-major: all r₀ values first, then r₁, and so on.
     *   This results in a flat array of `t × input_size` Zq values.
     *
     * For PolynomialRing<Zq, d>:
     *   The input is a vector of `input_size` polynomials P(x) = ∑ aᵢ·xⁱ,
     *   where each coefficient aᵢ ∈ Zq.
     *
     *   Each coefficient is decomposed independently:
     *
     *     aᵢ = rᵢ₀ + b·rᵢ₁ + ... + b^{t−1}·rᵢ_{t−1}
     *
     *   The output is a vector of `t × input_size` polynomials.
     *   Each group of `input_size` polynomials corresponds to one digit:
     *     - Rows-0 contains digit-0 polynomials (P₀(x) for all inputs)
     *     - Rows-1 contains digit-1 polynomials (P₁(x)), etc.
     *
     *   This layout is **digit-major**, meaning each row encodes the same digit across all input polynomials.
     *
     * @tparam T             Element type (`Zq` or `PolynomialRing<Zq, d>`)
     * @param input          Pointer to input elements.
     * @param input_size     Number of input elements (Zq scalars or polynomials).
     * @param base           Decomposition base `b` (must be ≥ 2).
     * @param config         Device execution and memory layout settings.
     * @param output         Output buffer: must hold `input_size × digits` elements.
     * @param output_size    Number of output elements (should be `input_size × digits`).
     * @return               eIcicleError indicating success or failure.
     */
    template <typename T>
    eIcicleError decompose(
      const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size);

    /**
     * @brief Recompose elements from their balanced base-b digit representation.
     *
     * Reconstructs each element x ∈ T as:
     *
     *     x = ∑ rᵢ · bⁱ mod q
     *
     * Input layout must be digit-major (i.e., match the layout from `decompose()`).
     *
     * @tparam T             Element type (`Zq` or `PolynomialRing<Zq, d>`)
     * @param input          Pointer to input digits (Zq or polynomials), in digit-major order.
     * @param input_size     Total number of input digits (should be `output_size × digits`).
     * @param base           Decomposition base `b`.
     * @param config         Execution configuration.
     * @param output         Output buffer to store recomposed elements.
     * @param output_size    Number of elements to reconstruct.
     * @return               eIcicleError indicating success or failure.
     */
    template <typename T>
    eIcicleError recompose(
      const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size);

  } // namespace balanced_decomposition
} // namespace icicle