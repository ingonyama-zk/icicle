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
    CpuSumcheckBackend(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config)
        : SumcheckBackend<F>(claimed_sum, std::move(transcript_config))
    {
    }

    eIcicleError get_proof(
      const std::vector<std::shared_ptr<std::vector<F>>>& mle_polynomials,
      const CombineFunction<F>& combine_function,
      const SumCheckConfig& config,
      SumCheckProof<F>& sumcheck_proof /*out*/) const override
    {
      return eIcicleError::API_NOT_IMPLEMENTED;
    }

    F get_alpha(std::vector<F>& round_polynomial) override { return F::zero(); }
  };

} // namespace icicle
