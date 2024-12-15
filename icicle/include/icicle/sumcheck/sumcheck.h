#pragma once

#include "icicle/errors.h"
#include "icicle/program/returning_value_program.h"
#include "icicle/sumcheck/sumcheck_proof.h"
#include "icicle/sumcheck/sumcheck_config.h"
#include "icicle/sumcheck/sumcheck_transcript_config.h"
#include "icicle/backend/sumcheck_backend.h"
namespace icicle {
  template <typename F>
  class Sumcheck;

  /**
   * @brief Static factory method to create a Sumcheck instance.
   *
   * @param claimed_sum The total sum of the values of a multivariate polynomial f(x₁, x₂, ..., xₖ)
   * when evaluated over all possible Boolean input combinations
   * @param transcript_config Configuration for encoding and hashing prover messages.
   * @return A Sumcheck object initialized with the specified backend.
   */
  template <typename F>
  Sumcheck<F> create_sumcheck(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config);

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
     * @brief Static factory method to create a Sumcheck instance.
     *
     * @param claimed_sum The total sum of the values of a multivariate polynomial f(x₁, x₂, ..., xₖ)
     * when evaluated over all possible Boolean input combinations
     * @param transcript_config Configuration for encoding and hashing prover messages.
     * @return A Sumcheck object initialized with the specified backend.
     */
    static Sumcheck<F> create(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config)
    {
      return create_sumcheck(claimed_sum, std::move(transcript_config));
    }

    /**
     * @brief Constructor for the Sumcheck class.
     * @param backend Shared pointer to the backend responsible for Sumcheck operations.
     */
    explicit Sumcheck(std::shared_ptr<SumcheckBackend<F>> backend) : m_backend{std::move(backend)} {}

    /**
     * @brief Calculate the sumcheck based on the inputs and retrieve the Sumcheck proof.
     * @param input_polynomials a vector of MLE polynomials to process
     * @param combine_function a program that define how to fold all MLS polynomials into the round polynomial.
     * @param config Configuration for the Sumcheck operation.
     * @param sumcheck_proof Reference to the SumCheckProof object where all round polynomials will be stored.
     * @return Error code of type eIcicleError.
     */
    eIcicleError get_proof(
      const std::vector<std::vector<F>*>& input_polynomials,
      const CombineFunction<F>& combine_function,
      SumCheckConfig& config,
      SumCheckProof<F>& sumcheck_proof /*out*/) const
    {
      return m_backend->get_proof(input_polynomials, combine_function, config, sumcheck_proof);
    }

    /**
     * @brief Verify an element against the Sumcheck round polynomial.
     * @param sumcheck_proof The SumCheckProof object includes the round polynomials.
     * @param valid output valid bit. True if the Proof is valid, false otherwise.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    eIcicleError verify(SumCheckProof<F>& sumcheck_proof, bool& valid /*out*/)
    {
      const int nof_rounds = sumcheck_proof.get_nof_round_polynomials();
      // verify that the sum of round_polynomial-0 is the clamed_sum
      const std::vector<F>& round_poly_0 = sumcheck_proof.get_round_polynomial(0);
      F round_poly_0_sum = round_poly_0[0];
      for (int round_idx = 1; round_idx < nof_rounds - 1; round_idx++) {
        round_poly_0_sum = round_poly_0_sum + round_poly_0[round_idx];
      }
      const F& claimed_sum = m_backend->get_claimed_sum();
      if (round_poly_0_sum != claimed_sum) {
        valid = false;
        ICICLE_LOG_ERROR << "verification failed: sum of round polynomial 0 (" << round_poly_0_sum
                         << ") != claimed_sum(" << claimed_sum << ")";
        return eIcicleError::SUCCESS;
      }

      for (int round_idx = 0; round_idx < nof_rounds - 1; round_idx++) {
        const std::vector<F>& round_poly = sumcheck_proof.get_round_polynomial(round_idx);
        F alpha = m_backend->get_alpha(round_poly);
        F alpha_value = lagrange_interpolation(round_poly, alpha);
        const std::vector<F>& next_round_poly = sumcheck_proof.get_round_polynomial(round_idx + 1);
        F expected_alpha_value = next_round_poly[0] + next_round_poly[1];
        if (alpha_value != expected_alpha_value) {
          valid = false;
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
    F lagrange_interpolation(const std::vector<F>& poly_evaluations, const F& x)
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