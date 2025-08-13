#include "icicle/backend/sumcheck_backend.h"
#include "cpu_sumcheck.h"

using namespace field_config;

namespace icicle {

  template <typename F>
  eIcicleError cpu_create_sumcheck_backend(const Device& device, std::shared_ptr<SumcheckBackend<F>>& backend /*OUT*/)
  {
    backend = std::make_shared<CpuSumcheckBackend<F>>();
    return eIcicleError::SUCCESS;
  }

  REGISTER_SUMCHECK_FACTORY_BACKEND("CPU", cpu_create_sumcheck_backend<scalar_t>);

#ifdef EXT_FIELD
  REGISTER_SUMCHECK_EXT_FIELD_BACKEND("CPU", cpu_create_sumcheck_backend<extension_t>);
#endif // EXT_FIELD

} // namespace icicle