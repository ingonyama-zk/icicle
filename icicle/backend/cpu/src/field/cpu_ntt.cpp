#include "cpu_ntt.h"

using namespace field_config;
using namespace icicle;

eIcicleError
cpu_ntt_init_domain(const Device& device, const scalar_t& primitive_root, const NTTInitDomainConfig& config)
{
  auto err = ntt_cpu::CpuNttDomain<scalar_t>::cpu_ntt_init_domain(device, primitive_root, config);
  return err;
}

template <typename S = scalar_t>
eIcicleError cpu_ntt_release_domain(const Device& device, const S& dummy)
{
  auto err = ntt_cpu::CpuNttDomain<scalar_t>::cpu_ntt_release_domain(device);
  return err;
}

template <typename S = scalar_t>
eIcicleError cpu_get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou)
{
  auto err = ntt_cpu::CpuNttDomain<scalar_t>::get_root_of_unity_from_domain(device, logn, rou);
  return err;
}

template <typename S, typename E>
eIcicleError
cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir dir, const NTTConfig<S>& config, E* output)
{
  auto err = ntt_cpu::cpu_ntt<S, E>(device, input, size, dir, config, output);
  return err;
}

// template <typename S, typename E>
// eIcicleError
// cpu_ntt_ref(const Device& device, const E* input, uint64_t size, NTTDir dir, const NTTConfig<S>& config, E* output)
// {
//   auto err = ntt_cpu::cpu_ntt_ref<S, E>(device, input, size, dir, config, output);
//   return err;
// }

REGISTER_NTT_INIT_DOMAIN_BACKEND("CPU", (cpu_ntt_init_domain));
REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CPU", cpu_ntt_release_domain<scalar_t>);
REGISTER_NTT_GET_ROU_FROM_DOMAIN_BACKEND("CPU", cpu_get_root_of_unity_from_domain<scalar_t>);
REGISTER_NTT_BACKEND("CPU", (cpu_ntt<scalar_t, scalar_t>));

// REGISTER_NTT_INIT_DOMAIN_BACKEND("CPU_REF", (cpu_ntt_init_domain));
// REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CPU_REF", cpu_ntt_release_domain<scalar_t>);
// REGISTER_NTT_GET_ROU_FROM_DOMAIN_BACKEND("CPU_REF", cpu_get_root_of_unity_from_domain<scalar_t>);
// REGISTER_NTT_BACKEND("CPU_REF", (cpu_ntt_ref<scalar_t, scalar_t>));

#ifdef EXT_FIELD
REGISTER_NTT_EXT_FIELD_BACKEND("CPU", (cpu_ntt<scalar_t, extension_t>));
#endif // EXT_FIELD
