#pragma once

#include "icicle/errors.h"
#include "icicle/program/returning_value_program.h"
#include "icicle/sumcheck/sumcheck_proof.h"
#include "icicle/sumcheck/sumcheck_config.h"
#include "icicle/sumcheck/sumcheck_transcript_config.h"
#include "icicle/backend/sumcheck_backend.h"
#include "icicle/sumcheck/sumcheck_transcript.h"

namespace icicle {
  template <typename F>
  class Sumcheck;

  /**
   * @brief Static factory method to create a Sumcheck instance.
   * @return A Sumcheck object initialized with the specified backend.
   */
  template <typename F>
  Sumcheck<F> create_sumcheck();

  /**
   * @brief Class for performing Sumcheck operations.
   *
   * This class provides a high-level interface for building and managing Sumcheck. The underlying
   * logic for sumcheck operations, such as building  and verifying, is delegated to the
   * backend, which may be device-specific (e.g., CPU, GPU).
   *
   * @tparam F The field type used in the Sumcheck protocol.
   */
  template <typename F>
  class Sumcheck
  {
  public:
    /**
     * @brief Constructor for the Sumcheck class.
     * @param backend Shared pointer to the backend responsible for Sumcheck operations.
     */
    explicit Sumcheck(std::shared_ptr<SumcheckBackend<F>> backend) : m_backend{std::move(backend)} {}

    /**
     * @brief Calculate the sumcheck based on the inputs and retrieve the Sumcheck proof.
     * @param mle_polynomials a vector of MLE polynomials to process.
     * F(X_1,X_2,X_3) = a_0 (1-X_1) (1-X_2) (1-X_3) + a_1 (1-X_1)(1-X_2) X_3 + a_2 (1-X_1) X_2 (1-X_3) +
     * a_3 (1-X_1) X_2 X_3 + a_4 X_1 (1-X_2) (1-X_3) + a_5 X_1 (1-X_2) X_3+ a_6 X_1 X_2 (1-X_3) + a_7 X_1 X_2 X_3
     * @param mle_polynomial_size the size of each MLE polynomial
     * @param claimed_sum The total sum of the values of a multivariate polynomial f(x₁, x₂, ..., xₖ)
     * when evaluated over all possible Boolean input combinations
     * @param combine_function a program that define how to fold all MLS polynomials into the round polynomial.
     * @param transcript_config Configuration for encoding and hashing prover messages.
     * @param sumcheck_config Configuration for the Sumcheck operation.
     * @param sumcheck_proof Reference to the SumcheckProof object where all round polynomials will be stored.
     * @return Error code of type eIcicleError.
     */
    eIcicleError get_proof(
      const std::vector<F*>& mle_polynomials,
      const uint64_t mle_polynomial_size,
      const F& claimed_sum,
      const CombineFunction<F>& combine_function,
      SumcheckTranscriptConfig<F>&& transcript_config,
      const SumcheckConfig& sumcheck_config,
      SumcheckProof<F>& sumcheck_proof /*out*/) const
    {
      return m_backend->get_proof(
        mle_polynomials, mle_polynomial_size, claimed_sum, combine_function, std::move(transcript_config),
        sumcheck_config, sumcheck_proof);
    }

    /**
     * @brief Verify an element against the Sumcheck round polynomial.
     * First round polynomial is verified against the claimed sum.
     * Last round polynomial is not verified.
     * @param sumcheck_proof The SumcheckProof object includes the round polynomials.
     * @param valid output valid bit. True if the Proof is valid, false otherwise.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    eIcicleError verify(
      const SumcheckProof<F>& sumcheck_proof,
      const F& claimed_sum,
      SumcheckTranscriptConfig<F>&& transcript_config,
      bool& valid /*out*/)
    {
      valid = false;
      const int nof_rounds = sumcheck_proof.get_nof_round_polynomials();
      const std::vector<F>& round_poly_0 = sumcheck_proof.get_const_round_polynomial(0);
      const uint32_t combine_function_poly_degree = round_poly_0.size() - 1;

      // verify that the sum of round_polynomial-0 is the clamed_sum
      F round_poly_0_sum = round_poly_0[0] + round_poly_0[1];

      if (round_poly_0_sum != claimed_sum) {
        valid = false;
        ICICLE_LOG_ERROR << "verification failed: sum of round polynomial 0 (" << round_poly_0_sum
                         << ") != claimed_sum(" << claimed_sum << ")";
        return eIcicleError::SUCCESS;
      }

      // create sumcheck_transcript for the Fiat-Shamir
      SumcheckTranscript sumcheck_transcript(
        claimed_sum, nof_rounds, combine_function_poly_degree, std::move(transcript_config));

      for (int round_idx = 0; round_idx < nof_rounds - 1; round_idx++) {
        const std::vector<F>& round_poly = sumcheck_proof.get_const_round_polynomial(round_idx);
        const F alpha = sumcheck_transcript.get_alpha(round_poly);
        const F alpha_value = lagrange_interpolation(round_poly, alpha);
        const std::vector<F>& next_round_poly = sumcheck_proof.get_const_round_polynomial(round_idx + 1);
        F expected_alpha_value = next_round_poly[0] + next_round_poly[1];
        if (alpha_value != expected_alpha_value) {
          ICICLE_LOG_ERROR << "verification failed on round: " << round_idx;
          return eIcicleError::SUCCESS;
        }
      }
      valid = true;
      return eIcicleError::SUCCESS;
    }

  private:
    std::shared_ptr<SumcheckBackend<F>>
      m_backend; ///< Shared pointer to the backend responsible for Sumcheck operations.

    // Receive the polynomial in evaluation on x=0,1,2...
    // return the evaluation of the polynomial at x
    F lagrange_interpolation(const std::vector<F>& poly_evaluations, const F& x) const
    {
      uint poly_degree = poly_evaluations.size();
      F result = F::zero();

      // For each coefficient we want to compute
      for (uint i = 0; i < poly_degree; ++i) {
        // Compute the i-th coefficient
        F numerator = poly_evaluations[i];
        F denumerator = F::one();

        // Use Lagrange interpolation formula
        const F i_field = F::from(i);
        for (uint j = 0; j < poly_degree; ++j) {
          if (j != i) {
            const F j_field = F::from(j);
            numerator = numerator * (x - j_field);
            denumerator = denumerator * (i_field - j_field);
          }
        }
        result = result + (numerator * F::inverse(denumerator));
      }
      return result;
    }
  };

} // namespace icicle