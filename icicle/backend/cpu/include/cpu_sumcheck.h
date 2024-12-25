#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"
#include "cpu_sumcheck_transcript.h"
#include "cpu_program_executor.h"
#include "icicle/backend/sumcheck_backend.h"

namespace icicle {






  template <typename F>
  class CpuSumcheckBackend : public SumcheckBackend<F>
  {
  public:
    CpuSumcheckBackend(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config)
        : SumcheckBackend<F>(claimed_sum, std::move(transcript_config))
    {
    }

    eIcicleError get_proof(
      const std::vector<F*>& mle_polynomials,
      const uint64_t mle_polynomial_size,
      const CombineFunction<F>& combine_function,
      const SumCheckConfig& config,
      SumCheckProof<F>& sumcheck_proof /*out*/) const override
    {
      // generate a program executor for the combine fumction
      CpuProgramExecutor program_executor(program);

      // calc the number of rounds = log2(poly_size)
      const int nof_rounds = std::log2(mle_polynomial_size);
      
      // run round 0 and update the proof
      calc_round_polynomial(mle_polynomials);
      for (int round_idx=1; round_idx < nof_rounds; ++round_idx) {
        // calculate alpha for the next round
        S alpha = fiat_shamir();

        // run the next round and update the proof
        calc_round_polynomial(m_folded_polynomials)
      }
      return eIcicleError::SUCCESS;
    }

    F get_alpha(std::vector<F>& round_polynomial) override { return F::zero(); }


  private:
    void calc_round_polynomial(const std::vector<F*>& input_polynomials, bool fold, S& alpha) {
      const uint64_t polynomial_size = fold ? input_polynomials[0].size()/4 : input_polynomials[0].size()/2;
      const uint nof_polynomials = input_polynomials.size();
      const uint round_poly_degree = m_combine_prog.m_poly_degree;
      m_round_polynomial.resize(round_poly_degree+1); 
      TODO:: init m_round_polynomial to zero;

      // init m_program_executor input pointers
      vector<std::vector<S> combine_func_inputs(m_combine_prog.m_nof_inputs);
      for (int poly_idx=0; i<nof_polynomials; ++poly_idx) {
        m_program_executor.m_variable_ptrs[poly_idx] = &(combine_func_inputs[poly_idx]);
      }
      // init m_program_executor output pointer
      S combine_func_result;
      m_program_executor.m_variable_ptrs[nof_polynomials] = combine_func_result;

      const one_minus_alpha = (1-alpha);
      // run over all elements in all polynomials
      for (int element_idx=0; i<polynomial_size; ++element_idx) {
        // init combine_func_inputs for k=0
        for (int poly_idx=0; i<nof_polynomials; ++poly_idx) {
          // when k=0, take the element i
          if (fold) {
            m_folded_polynomials[element_idx] = 
              one_minus_alpha * input_polynomials[element_idx] + 
              alpha *           input_polynomials[element_idx + 2*polynomial_size];
            combine_func_inputs[poly_idx] = m_folded_polynomials[element_idx];
          }
          else {
            combine_func_inputs[poly_idx] = input_polynomials[poly_idx][element_idx];
          }
        }
        
        for (int k=0; k <= round_poly_degree; ++k) {
          // execute the combine functions and append to the round polynomial
          m_program_executor.execute();
          m_round_polynomial[k] += combine_func_result;

          // if this is not the last k
          if (k < round_poly_degree) {
            // update the combine program inputs for the next k
            for (int poly_idx=0; i<nof_polynomials; ++poly_idx) {
              if (fold) {
                m_folded_polynomials[poly_idx] = 
                  m_folded_polynomials[poly_idx]
                  - m_folded_polynomials[poly_idx][element_idx]
                  + m_folded_polynomials[poly_idx][element_idx+polynomial_size];
              }
              else {
                combine_func_inputs[poly_idx] = 
                  combine_func_inputs[poly_idx]
                  - input_polynomials[poly_idx][element_idx]
                  + input_polynomials[poly_idx][element_idx+polynomial_size];
              }
            }
          }
        }
      }
    }



  };

} // namespace icicle
