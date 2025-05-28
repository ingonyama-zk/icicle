#pragma once

namespace icicle {

  /**
   * @brief Polynomial ring over a finite ring/field F in the coefficient domain.
   *
   * Represents polynomials of the form:
   *     a(x) = a₀ + a₁·x + a₂·x² + ... + a_{d−1}·x^{d−1}
   *
   * Elements are stored as F coefficients, where F is a base ring or field,
   * and `degree` is the number of coefficients (typically a power of 2).
   *
   * Conceptually: PolynomialRing<F, d> ≡ F[x] / (x^d + 1)
   */
  template <typename F, uint32_t degree>
  class PolynomialRing
  {
  public:
    using Base = F;                       ///< Base ring or field type
    static constexpr uint32_t d = degree; ///< Degree of the polynomial (number of coefficients)

    F coeffs[d]; ///< Coefficients in standard (coefficient) domain
  };

  /**
   * @brief Polynomial ring over a finite ring/field F in the NTT (evaluation) domain.
   *
   * Represents polynomials after applying the Number Theoretic Transform (NTT),
   * which converts a polynomial from the coefficient domain to the evaluation domain.
   *
   * This structure is mathematically equivalent to PolynomialRing<F, d>, but
   * the values are evaluations at NTT roots of unity. Enables efficient
   * multiplication via pointwise operations.
   */
  template <typename F, uint32_t degree>
  class NTTPolynomialRing
  {
  public:
    using Base = F;                       ///< Base ring or field type
    static constexpr uint32_t d = degree; ///< Degree of the polynomial (number of evaluations)
    F evaluations[d];                     ///< Evaluations at NTT domain points
  };

} // namespace icicle