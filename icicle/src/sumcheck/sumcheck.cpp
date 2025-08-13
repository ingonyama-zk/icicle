#include "icicle/errors.h"
#include "icicle/sumcheck/sumcheck.h"
#include "icicle/backend/sumcheck_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(SumcheckDispatcher, sumcheck_factory, SumcheckFactoryImpl<scalar_t>);

  template <>
  Sumcheck<scalar_t> create_sumcheck()
  {
    std::shared_ptr<SumcheckBackend<scalar_t>> backend;
    ICICLE_CHECK(SumcheckDispatcher::execute(backend));
    Sumcheck<scalar_t> sumcheck{backend};
    return sumcheck;
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(SumcheckExtDispatcher, extension_sumcheck_factory, SumcheckExtFieldImpl<extension_t>);

  template <>
  Sumcheck<extension_t> create_sumcheck()
  {
    std::shared_ptr<SumcheckBackend<extension_t>> backend;
    ICICLE_CHECK(SumcheckExtDispatcher::execute(backend));
    Sumcheck<extension_t> sumcheck{backend};
    return sumcheck;
  }
#endif // EXT_FIELD

} // namespace icicle