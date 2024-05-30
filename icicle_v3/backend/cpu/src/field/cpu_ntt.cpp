
#include "icicle/ntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

template <typename S = scalar_t, typename E = scalar_t>
eIcicleError cpu_ntt(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
{
  std::cerr << "cpu_ntt() not implemented" << std::endl;
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_NTT_BACKEND("CPU", (cpu_ntt<scalar_t, scalar_t>));

template <typename S = scalar_t>
eIcicleError cpu_ntt_init_domain(const Device& device, const S& primitive_root, const ConfigExtension& config)
{
  std::cerr << "cpu_ntt_init_domain() not implemented" << std::endl;
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_NTT_INIT_DOMAIN_BACKEND("CPU", (cpu_ntt_init_domain<scalar_t>));

eIcicleError cpu_ntt_release_domain(const Device& device)
{
  std::cerr << "cpu_ntt_release_domain() not implemented" << std::endl;
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CPU", cpu_ntt_release_domain);