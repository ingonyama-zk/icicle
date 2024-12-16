#include "icicle/backend/sumcheck_backend.h"
#include "cpu_sumcheck.h"

namespace icicle {

  template <typename F>
  eIcicleError create_sumcheck_backend(
    const Device& device,
    const F& claimed_sum, 
    SumcheckTranscriptConfig<F>&& transcript_config,
    std::shared_ptr<SumcheckBackend<F>>& backend)
  {
    backend = std::make_shared<SumcheckBackend>(claimed_sum, std::move(transcript_config));
    return eIcicleError::SUCCESS;
  }

  REGISTER_SUMCHECK_FACTORY_BACKEND("CPU", create_sumcheck_backend<scalar_t>);

}