#include "icicle/errors.h"
#include "icicle/sumcheck/sumcheck.h"
#include "icicle/backend/sumcheck_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(SumcheckDispatcher, sumcheck_factory, SumcheckFactoryImpl);

  template <>
  Sumcheck<F> create_sumcheck(F& claimed_sum, const SumcheckTranscriptConfig<F>&& transcript_config)
  {
    std::shared_ptr<SumcheckBackend> backend;
    ICICLE_CHECK(SumcheckDispatcher::execute(claimed_sum, std::move(transcript_config)));
    Sumcheck sumcheck{backend};
    return sumcheck;
  }

} // namespace icicle