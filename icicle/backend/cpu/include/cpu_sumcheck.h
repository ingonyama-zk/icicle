#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"
#include "icicle/backend/sumcheck_backend.h"

namespace icicle {

template <typename F>
class CpuSumcheckBackend : public SumcheckBackend<F>
{
  public:
    CpuSumcheckBackend(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config) :
    SumcheckBackend<F>(claimed_sum, std::move(transcript_config))
    {}

    virtual eIcicleError get_proof(
      const std::vector< std::vector<F>* >& input_polynomials, 
      const CombineFunction<F>& combine_function, 
      SumCheckConfig& config,
      SumCheckProof<F>& sumcheck_proof /*out*/) override{

        
      }

    virtual F get_alpha(std::vector< F >& round_polynomial) override {
      const std::vector<std::byte>& round_poly_label = this->m_transcript_config.m_round_poly_label();
      std::vector<std::byte> hash_input;

      // hash hash_input and return alpha
      std::vector<std::byte> hash_result(this->m_transcript_config.hasher.output_size());
      this->m_transcript_config.hasher.hash(hash_input.data(), hash_input.size(), this->m_config, hash_result.data());
      this->m_prev_alpha = F::reduce(hash_result.data()); //TODO fix that
      return this->m_prev_alpha;
    } 

  };

}

