#include "icicle/errors.h"
#include "icicle/sumcheck/sumcheck.h"
#include "icicle/backend/sumcheck_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(SumcheckDispatcher, sumcheck_factory, SumcheckFactoryImpl<scalar_t>);

  template <>
  Sumcheck<scalar_t>
  create_sumcheck(const scalar_t& claimed_sum, SumcheckTranscriptConfig<scalar_t>&& transcript_config)
  {
    std::shared_ptr<SumcheckBackend<scalar_t>> backend;
    ICICLE_CHECK(SumcheckDispatcher::execute(claimed_sum, std::move(transcript_config), backend));
    Sumcheck<scalar_t> sumcheck{backend};
    return sumcheck;
  }
} // namespace icicle