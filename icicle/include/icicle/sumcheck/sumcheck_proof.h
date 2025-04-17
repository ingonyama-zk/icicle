#pragma once

#include <vector>
#include <memory>
#include "icicle/serialization.h"
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
  class SumcheckProof : public Serializable
  {
  public:
    // Constructor for creating a new proof
    SumcheckProof() {}

    // Constructor for a verifier to use given a prover's proof
    SumcheckProof(std::vector<std::vector<S>> polys) : m_round_polynomials(polys) {}

    // Init the round polynomial values for the problem
    void init(int nof_round_polynomials, int round_polynomial_degree)
    {
      if (nof_round_polynomials == 0) {
        ICICLE_LOG_ERROR << "Number of round polynomials(" << nof_round_polynomials << ") in the proof must be > 0";
      }
      m_round_polynomials.resize(nof_round_polynomials, std::vector<S>(round_polynomial_degree + 1, S::zero()));
    }

    // return a reference to the round polynomial generated at round # round_polynomial_idx
    const std::vector<S>& get_const_round_polynomial(int round_polynomial_idx) const
    {
      return m_round_polynomials[round_polynomial_idx];
    }

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

    eIcicleError serialized_size(size_t& size) const override
    {
      size = sizeof(size_t); // nof_round_polynomials
      for (const auto& round_poly : m_round_polynomials) {
        size += sizeof(size_t); // nested vector size
        size += round_poly.size() * sizeof(S);
      }
      return eIcicleError::SUCCESS;
    }

    eIcicleError serialize(std::byte*& buffer, size_t& buffer_length) const override
    {
      size_t nof_round_polynomials = m_round_polynomials.size();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &nof_round_polynomials, sizeof(size_t)));
      for (const auto& round_poly : m_round_polynomials) {
        size_t round_poly_size = round_poly.size();
        ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &round_poly_size, sizeof(size_t)));
        ICICLE_CHECK_IF_RETURN(
          memcpy_shift_destination(buffer, buffer_length, round_poly.data(), round_poly_size * sizeof(S)));
      }
      return eIcicleError::SUCCESS;
    }

    eIcicleError deserialize(std::byte*& buffer, size_t& buffer_length) override
    {
      size_t nof_round_polynomials;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&nof_round_polynomials, buffer_length, buffer, sizeof(size_t)));

      m_round_polynomials.resize(nof_round_polynomials);
      for (size_t i = 0; i < nof_round_polynomials; ++i) {
        size_t round_poly_size;
        ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&round_poly_size, buffer_length, buffer, sizeof(size_t)));

        size_t byte_size = round_poly_size * sizeof(S);
        m_round_polynomials[i].resize(round_poly_size);
        ICICLE_CHECK_IF_RETURN(memcpy_shift_source(m_round_polynomials[i].data(), buffer_length, buffer, byte_size));
      }

      return eIcicleError::SUCCESS;
    }
  };

} // namespace icicle