#pragma once

namespace icicle {

  /**
   * @brief Polynomial ring over a finite field or ring F, interpreted as either
   *        coefficient or evaluation (NTT) domain.
   *
   * Represents a polynomial in one of two domains:
   *   - Coefficient domain:
   *       a(x) = a₀ + a₁·x + a₂·x² + ... + a_{d−1}·x^{d−1}
   *   - NTT (evaluation) domain:
   *       â = [â₀, â₁, ..., â_{d−1}], where âᵢ = a(ωᵢ)
   *
   * The interpretation is left to the user. No distinction is enforced at type level.
   *
   * Algebraically: F[x] / (x^d + 1)
   */
  template <typename F, uint32_t degree>
  class PolynomialRing
  {
  public:
    using Base = F;                       ///< Base field or ring type
    static constexpr uint32_t d = degree; ///< Number of coefficients or evaluations

    F values[d]; ///< Polynomial data (coeffs or NTT values)

    friend bool operator==(const PolynomialRing& lhs, const PolynomialRing& rhs)
    {
      return std::memcmp(lhs.values, rhs.values, sizeof(F) * d) == 0;
    }
    friend bool operator!=(const PolynomialRing& lhs, const PolynomialRing& rhs) { return !(lhs == rhs); }
  };

} // namespace icicle