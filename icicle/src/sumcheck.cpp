// #include "icicle/errors.h"
// #include "icicle/sumcheck/sumcheck.h"
// #include "icicle/backend/sumcheck_backend.h"
// #include "icicle/dispatcher.h"

// namespace icicle {

//   ICICLE_DISPATCHER_INST(SumcheckDispatcher, sumcheck_factory, SumcheckFactoryImpl);

//   template <>
//   Sumcheck<scalar_t>
//   create_sumcheck(scalar_t& claimed_sum, const SumcheckTranscriptConfig<scalar_t>&& transcript_config)
//   {
//     std::shared_ptr<SumcheckBackend> backend;
//     ICICLE_CHECK(SumcheckDispatcher::execute(claimed_sum, std::move(transcript_config)));
//     Sumcheck sumcheck{backend};
//     return sumcheck;
//   }

// } // namespace icicle