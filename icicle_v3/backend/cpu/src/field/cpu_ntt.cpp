
#include "icicle/ntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "ntt.template"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

REGISTER_NTT_BACKEND("CPU", (cpu_ntt<scalar_t, scalar_t>));
#ifdef EXT_FIELD
REGISTER_NTT_EXT_FIELD_BACKEND("CPU", (cpu_ntt<scalar_t, extension_t>));
#endif // EXT_FIELD

template <typename S = scalar_t>
eIcicleError cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config)
{
  ICICLE_LOG_ERROR << "cpu_ntt_init_domain() not implemented";
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_NTT_INIT_DOMAIN_BACKEND("CPU", (cpu_ntt_init_domain<scalar_t>));

template <typename S = scalar_t>
eIcicleError cpu_ntt_release_domain(const Device& device, const S& dummy)
{
  ICICLE_LOG_ERROR << "cpu_ntt_release_domain() not implemented";
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CPU", cpu_ntt_release_domain<scalar_t>);