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
      const SumCheckConfig& config,
      SumCheckProof<F>& sumcheck_proof /*out*/) override
    {
      const int nof_mle_poly = mle_polynomials.size();
      std::vector<F*> folded_mle_polynomials(nof_mle_poly);
      for (auto& mle_poly_ptr : folded_mle_polynomials) {
        mle_poly_ptr = new F[mle_polynomial_size / 2];
      }

      // Check that the size of the the proof feet the size of the mle polynomials.
      const uint32_t nof_rounds = sumcheck_proof.get_nof_round_polynomials();
      if (std::log2(mle_polynomial_size) != nof_rounds) {
        ICICLE_LOG_ERROR << "Sumcheck proof size(" << nof_rounds << ") should be log of the mle polynomial size("
                         << mle_polynomial_size << ")";
        return eIcicleError::INVALID_ARGUMENT;
      }
      // check that the combine function has a legal polynomial degree
      int poly_degree = combine_function.get_polynomial_degee();
      if (poly_degree < 0) {
        ICICLE_LOG_ERROR << "Illegal polynomial degree (" << poly_degree << ") for provided combine function";
        return eIcicleError::INVALID_ARGUMENT;
      }
      reset(nof_rounds, uint32_t(poly_degree));

      // reset the sumcheck proof to accumulate
      sumcheck_proof.reset();

      // generate a program executor for the combine function
      CpuProgramExecutor program_executor(combine_function);

      // calc the number of rounds = log2(poly_size)

      int cur_mle_polynomial_size = mle_polynomial_size;

      for (int round_idx = 0; round_idx < nof_rounds; ++round_idx) {
        const std::vector<F*>& in_mle_polynomials = (round_idx == 0) ? mle_polynomials : folded_mle_polynomials;
        std::vector<F>& round_polynomial = sumcheck_proof.get_round_polynomial(round_idx);

        // run the next round and update the proof
        build_round_polynomial(in_mle_polynomials, cur_mle_polynomial_size, program_executor, round_polynomial);

        if (round_idx + 1 < nof_rounds) {
          // calculate alpha for the next round
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

    // Reset the sumcheck transcript with e
    void reset(const uint32_t num_vars, const uint32_t poly_degree) override
    {
      m_cpu_sumcheck_transcript.reset(num_vars, poly_degree);
    }

  private:
    // members
    CpuSumCheckTranscript<F> m_cpu_sumcheck_transcript; // Generates alpha for the next round (Fial-Shamir)

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

    void fold_mle_polynomials(
      const F& alpha,
      int& mle_polynomial_size,
      const std::vector<F*>& in_mle_polynomials,
      std::vector<F*>& folded_mle_polynomials)
    {
      const int nof_polynomials = in_mle_polynomials.size();
      const F one_minus_alpha = F::one() - alpha;
      mle_polynomial_size >>= 1;

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

    //   void calc_round_polynomial(const std::vector<F*>& input_polynomials, bool fold, S& alpha) {
    //     const uint64_t polynomial_size = fold ? input_polynomials[0].size()/4 : input_polynomials[0].size()/2;
    //     const uint nof_polynomials = input_polynomials.size();
    //     const uint round_poly_degree = m_combine_prog.m_poly_degree;
    //     m_round_polynomial.resize(round_poly_degree+1);
    //     TODO:: init m_round_polynomial to zero;

    //     // init m_program_executor input pointers
    //     vector<std::vector<S> combine_func_inputs(m_combine_prog.m_nof_inputs);
    //     for (int poly_idx=0; i<nof_polynomials; ++poly_idx) {
    //       m_program_executor.m_variable_ptrs[poly_idx] = &(combine_func_inputs[poly_idx]);
    //     }
    //     // init m_program_executor output pointer
    //     S combine_func_result;
    //     m_program_executor.m_variable_ptrs[nof_polynomials] = combine_func_result;

    //     const one_minus_alpha = (1-alpha);
    //     // run over all elements in all polynomials
    //     for (int element_idx=0; i<polynomial_size; ++element_idx) {
    //       // init combine_func_inputs for k=0
    //       for (int poly_idx=0; i<nof_polynomials; ++poly_idx) {
    //         // when k=0, take the element i
    //         if (fold) {
    //           m_folded_polynomials[element_idx] =
    //             one_minus_alpha * input_polynomials[element_idx] +
    //             alpha *           input_polynomials[element_idx + 2*polynomial_size];
    //           combine_func_inputs[poly_idx] = m_folded_polynomials[element_idx];
    //         }
    //         else {
    //           combine_func_inputs[poly_idx] = input_polynomials[poly_idx][element_idx];
    //         }
    //       }

    //       for (int k=0; k <= round_poly_degree; ++k) {
    //         // execute the combine functions and append to the round polynomial
    //         m_program_executor.execute();
    //         m_round_polynomial[k] += combine_func_result;

    //         // if this is not the last k
    //         if (k < round_poly_degree) {
    //           // update the combine program inputs for the next k
    //           for (int poly_idx=0; i<nof_polynomials; ++poly_idx) {
    //             if (fold) {
    //               m_folded_polynomials[poly_idx] =
    //                 m_folded_polynomials[poly_idx]
    //                 - m_folded_polynomials[poly_idx][element_idx]
    //                 + m_folded_polynomials[poly_idx][element_idx+polynomial_size];
    //             }
    //             else {
    //               combine_func_inputs[poly_idx] =
    //                 combine_func_inputs[poly_idx]
    //                 - input_polynomials[poly_idx][element_idx]
    //                 + input_polynomials[poly_idx][element_idx+polynomial_size];
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
  };

} // namespace icicle
