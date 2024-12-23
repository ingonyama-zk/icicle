#pragma once

#include <vector>
#include <memory>

namespace icicle {

  /**
   * @brief Represents a sumcheck proof.
   *
   * This class encapsulates the sumcheck proof, contains the evaluations of the round polynomials
   * for each layer at the sumcheck proof.
   * Evaluations are at x = 0, 1, 2 ... K
   * Where K is the degree of the combiine function used at the sumcheck protocol.
   *
   * @tparam S Type of the field element (e.g., prime field or extension field elements).
   */

  template <typename S>
  class SumCheckProof
  {
  public:
    // Constructor
    SumCheckProof(uint nof_round_polynomials, uint round_polynomial_degree)
        : m_round_polynomials(nof_round_polynomials, std::vector<S>(round_polynomial_degree + 1))
    {
      if (nof_round_polynomials == 0) {
        ICICLE_LOG_ERROR << "Number of round polynomials(" << nof_round_polynomials << ") in the proof must be >0";
      }
    }

    // set the value of polynomial round_polynomial_idx at x = evaluation_idx
    void set_round_polynomial_value(int round_polynomial_idx, int evaluation_idx, S& value)
    {
      m_round_polynomials[round_polynomial_idx][evaluation_idx] = value;
    }

    // return a reference to the round polynomial generated at round # round_polynomial_idx
    const std::vector<S>& get_round_polynomial(int round_polynomial_idx) const
    {
      return m_round_polynomials[round_polynomial_idx];
    }

    uint get_nof_round_polynomial() const { return m_round_polynomials.size(); }
    uint get_round_polynomial_size() const { return m_round_polynomials[0].size() + 1; }

  private:
    std::vector<std::vector<S>> m_round_polynomials; // logN vectors of round_poly_degree elements
  };

} // namespace icicle