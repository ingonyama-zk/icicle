#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/sumcheck/sumcheck.h"

using namespace field_config;

namespace icicle {
  extern "C" {

  typedef Sumcheck<scalar_t> SumcheckHandle;

  SumcheckHandle* CONCAT_EXPAND(FIELD, sumcheck_create)()
  {
    ICICLE_LOG_INFO << "hello world";
    // TODO Yuval update params and implement
    return nullptr;
  }

  } // extern "C"

} // namespace icicle