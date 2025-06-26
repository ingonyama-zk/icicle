#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {
  namespace balanced_decomposition {

    /**
     * @brief Compute the number of digits required to represent any element
     *        using balanced base-b decomposition.
     *
     * Supports both:
     *   - Scalar rings (e.g., Zq)
     *   - Polynomial rings (e.g., Rq = Zq[x]/(X^d + 1), modeled as PolynomialRing<Zq, d>)
     *
     * The number of digits is computed per scalar coefficient (Zq), and applies
     * equally to each coefficient in a polynomial.
     *
     * Each Zq element is represented using digits in the range (-b/2, b/2], and
     * an extra digit is added when base > 2 to handle carry propagation.
     *
     * @tparam T    Element type. Must define `T::TLC == 2` and `T::get_modulus()`.
     *              Typically `Zq` or `PolynomialRing<Zq, d>`.
     * @param base  Decomposition base b (must be ≥ 2).
     * @return      Number of balanced digits required to represent any element of type T.
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
     * @brief Decomposes elements into balanced base-b digits.
     *
     * For scalars (T = Zq):
     *   Each element x ∈ Zq is decomposed as:
     *
     *     x ≡ r₀ + b·r₁ + b²·r₂ + ... + b^{t−1}·r_{t−1} mod q
     *
     *   The output layout is **element-major**, meaning all t digits of the first
     *   element are stored consecutively, followed by the digits of the second element, and so on.
     *
     * For polynomials (T = PolynomialRing<Zq, d>):
     *   Each coefficient a_j ∈ Zq of a polynomial P(x) is decomposed independently into t digits:
     *
     *     a_j = r_{j,0} + b·r_{j,1} + ... + b^{t−1}·r_{j,t−1}
     *
     *   The result is a sequence of t polynomials:
     *
     *     P(x) = P₀(x) + b·P₁(x) + b²·P₂(x) + ... + b^{t−1}·P_{t−1}(x)
     *
     *   The output layout is **digit-major**, meaning all first digits of all input polynomials
     *   come first (as polynomials), followed by all second digits, and so on.
     *
     * @tparam T             Element type (`Zq` or `PolynomialRing<Zq, d>`)
     * @param input          Pointer to input elements.
     * @param input_size     Number of input elements.
     * @param base           Decomposition base `b`.
     * @param config         Vectorization and device configuration.
     * @param output         Output buffer (must hold input_size × digits elements).
     * @param output_size    Number of output elements.
     * @return               eIcicleError indicating success or failure.
     */
    template <typename T>
    eIcicleError decompose(
      const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size);

    /**
     * @brief Recomposes elements from balanced base-b digits.
     *
     * For each element x ∈ T, reconstructs:
     *
     *     x = ∑ r_i · b^i mod q
     *
     * For PolynomialRing<Zq, d>, this applies to each coefficient independently.
     *
     * Input layout expectations:
     *   - For scalars (Zq): the digits are stored element-major (each element's digits in sequence).
     *   - For polynomials: the digits are stored digit-major (all P₀(x), then all P₁(x), ...).
     *
     * The layout must match that produced by `decompose()`.
     *
     * @tparam T             Element type (`Zq` or `PolynomialRing<Zq, d>`)
     * @param input          Pointer to flattened input digits.
     * @param input_size     Number of input elements (digits × input_size).
     * @param base           The base `b` used in decomposition.
     * @param config         Vectorization and device configuration.
     * @param output         Output buffer to store recomposed elements.
     * @param output_size    Number of elements to reconstruct.
     * @return               eIcicleError indicating success or failure.
     */
    template <typename T>
    eIcicleError recompose(
      const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size);

  } // namespace balanced_decomposition
} // namespace icicle