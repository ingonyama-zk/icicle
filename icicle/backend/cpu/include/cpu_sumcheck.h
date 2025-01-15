#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"
#include "cpu_sumcheck_transcript.h"
#include "cpu_program_executor.h"
#include "icicle/backend/sumcheck_backend.h"
#include "cpu_sumcheck_transcript.h"
namespace icicle {
  template <typename F>
  class CpuSumcheckBackend : public SumcheckBackend<F>
  {
  public:
    CpuSumcheckBackend(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config)
        : SumcheckBackend<F>(claimed_sum, std::move(transcript_config)),
          m_cpu_sumcheck_transcript(claimed_sum, std::move(transcript_config))
    {
    }

    // Calculate a proof for the mle polynomials
    eIcicleError get_proof(
      const std::vector<F*>& mle_polynomials,
      const uint64_t mle_polynomial_size,
      const CombineFunction<F>& combine_function,
      const SumcheckConfig& config,
      SumcheckProof<F>& sumcheck_proof /*out*/) override
    {
      if (config.use_extension_field) {
        ICICLE_LOG_ERROR << "SumcheckConfig::use_extension_field field = true is currently unsupported";
        return eIcicleError::INVALID_ARGUMENT;
      }
      // Allocate memory for the intermediate calculation: the folded mle polynomials
      const int nof_mle_poly = mle_polynomials.size();
      std::vector<F*> folded_mle_polynomials(nof_mle_poly); // folded mle_polynomials with the same format as inputs
      std::vector<F> folded_mle_polynomials_values(
        nof_mle_poly * mle_polynomial_size / 2); // folded_mle_polynomials data itself
      // init the folded_mle_polynomials pointers
      for (int mle_polynomial_idx = 0; mle_polynomial_idx < nof_mle_poly; mle_polynomial_idx++) {
        folded_mle_polynomials[mle_polynomial_idx] =
          &(folded_mle_polynomials_values[mle_polynomial_idx * mle_polynomial_size / 2]);
      }

      // Check that the size of the the proof feet the size of the mle polynomials.
      const uint32_t nof_rounds = std::log2(mle_polynomial_size);

      // check that the combine function has a legal polynomial degree
      int combine_function_poly_degree = combine_function.get_polynomial_degee();
      if (combine_function_poly_degree < 0) {
        ICICLE_LOG_ERROR << "Illegal polynomial degree (" << combine_function_poly_degree
                         << ") for provided combine function";
        return eIcicleError::INVALID_ARGUMENT;
      }

      reset_transcript(nof_rounds, uint32_t(combine_function_poly_degree)); // reset the transcript for the Fiat-Shamir
      sumcheck_proof.init(
        nof_rounds,
        uint32_t(combine_function_poly_degree)); // reset the sumcheck proof to accumulate the round polynomials

      // generate a program executor for the combine function
      CpuProgramExecutor program_executor(combine_function);

      // run log2(poly_size) rounds
      int cur_mle_polynomial_size = mle_polynomial_size;
      for (int round_idx = 0; round_idx < nof_rounds; ++round_idx) {
        // For the first round work on the input mle_polynomials, else work on the folded
        const std::vector<F*>& in_mle_polynomials = (round_idx == 0) ? mle_polynomials : folded_mle_polynomials;
        std::vector<F>& round_polynomial = sumcheck_proof.get_round_polynomial(round_idx);

        // build round polynomial and update the proof
        build_round_polynomial(in_mle_polynomials, cur_mle_polynomial_size, program_executor, round_polynomial);

        // if its not the last round, calculate alpha and fold the mle polynomials
        if (round_idx + 1 < nof_rounds) {
          F alpha = get_alpha(round_polynomial);
          fold_mle_polynomials(alpha, cur_mle_polynomial_size, in_mle_polynomials, folded_mle_polynomials);
        }
      }
      return eIcicleError::SUCCESS;
    }

    // calculate alpha for the next round based on the round_polynomial of the current round
    F get_alpha(std::vector<F>& round_polynomial) override
    {
      return m_cpu_sumcheck_transcript.get_alpha(round_polynomial);
    }

    // Reset the sumcheck transcript before a new proof generation or verification
    void reset_transcript(const uint32_t mle_polynomial_size, const uint32_t combine_function_poly_degree) override
    {
      m_cpu_sumcheck_transcript.reset(mle_polynomial_size, combine_function_poly_degree);
    }

  private:
    // members
    CpuSumcheckTranscript<F> m_cpu_sumcheck_transcript; // Generates alpha for the next round (Fial-Shamir)

    void build_round_polynomial(
      const std::vector<F*>& in_mle_polynomials,
      const int mle_polynomial_size,
      CpuProgramExecutor<F>& program_executor,
      std::vector<F>& round_polynomial)
    {
      // init program_executor input pointers
      const int nof_polynomials = in_mle_polynomials.size();
      std::vector<F> combine_func_inputs(nof_polynomials);
      for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
        program_executor.m_variable_ptrs[poly_idx] = &(combine_func_inputs[poly_idx]);
      }
      // init m_program_executor output pointer
      F combine_func_result;
      program_executor.m_variable_ptrs[nof_polynomials] = &combine_func_result;

      const int round_poly_size = round_polynomial.size();
      for (int element_idx = 0; element_idx < mle_polynomial_size / 2; ++element_idx) {
        for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
          combine_func_inputs[poly_idx] = in_mle_polynomials[poly_idx][element_idx];
        }
        for (int k = 0; k < round_poly_size; ++k) {
          // execute the combine functions and append to the round polynomial
          program_executor.execute();
          round_polynomial[k] = round_polynomial[k] + combine_func_result;

          // if this is not the last k
          if (k + 1 < round_poly_size) {
            // update the combine program inputs for the next k
            for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
              combine_func_inputs[poly_idx] = combine_func_inputs[poly_idx] -
                                              in_mle_polynomials[poly_idx][element_idx] +
                                              in_mle_polynomials[poly_idx][element_idx + mle_polynomial_size / 2];
            }
          }
        }
      }
    }

    // Fold the MLE polynomials based on alpha
    void fold_mle_polynomials(
      const F& alpha,
      int& mle_polynomial_size,
      const std::vector<F*>& in_mle_polynomials, // input
      std::vector<F*>& folded_mle_polynomials)   // output
    {
      const int nof_polynomials = in_mle_polynomials.size();
      const F one_minus_alpha = F::one() - alpha;
      mle_polynomial_size >>= 1; // update the mle_polynomial size to /2 det to folding

      // run over all elements in all polynomials
      for (int element_idx = 0; element_idx < mle_polynomial_size; ++element_idx) {
        // init combine_func_inputs for k=0
        for (int poly_idx = 0; poly_idx < nof_polynomials; ++poly_idx) {
          folded_mle_polynomials[poly_idx][element_idx] =
            one_minus_alpha * in_mle_polynomials[poly_idx][element_idx] +
            alpha * in_mle_polynomials[poly_idx][element_idx + mle_polynomial_size];
        }
      }
    }
  };

} // namespace icicle
