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
   * Where K is the degree of the combine function used at the sumcheck protocol.
   *
   * @tparam S Type of the field element (e.g., prime field or extension field elements).
   */

  template <typename S>
  class SumcheckProof
  {
  public:
    // Constructor
    SumcheckProof() {}

    // Init the round polynomial values for the problem
    void init(int nof_round_polynomials, int round_polynomial_degree)
    {
      if (nof_round_polynomials == 0) {
        ICICLE_LOG_ERROR << "Number of round polynomials(" << nof_round_polynomials << ") in the proof must be > 0";
      }
      m_round_polynomials.resize(nof_round_polynomials, std::vector<S>(round_polynomial_degree + 1, S::zero()));
    }

    // return a reference to the round polynomial generated at round # round_polynomial_idx
    const std::vector<S>& get_const_round_polynomial(int round_polynomial_idx) const { return m_round_polynomials[round_polynomial_idx]; }

    // return a reference to the round polynomial generated at round # round_polynomial_idx
    std::vector<S>& get_round_polynomial(int round_polynomial_idx) { return m_round_polynomials[round_polynomial_idx]; }

    uint get_nof_round_polynomials() const { return m_round_polynomials.size(); }
    uint get_round_polynomial_size() const { return m_round_polynomials[0].size() + 1; }

  private:
    std::vector<std::vector<S>> m_round_polynomials; // logN vectors of round_poly_degree elements

  public:
    // for debug
    void print_proof()
    {
      std::cout << "Sumcheck Proof :" << std::endl;
      for (int round_poly_i = 0; round_poly_i < m_round_polynomials.size(); round_poly_i++) {
        std::cout << "  Round polynomial " << round_poly_i << ":" << std::endl;
        for (auto& element : m_round_polynomials[round_poly_i]) {
          std::cout << "    " << element << std::endl;
        }
      }
    }
  };

} // namespace icicle