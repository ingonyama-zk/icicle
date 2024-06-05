#include "ntt_template.h"
#include "icicle/ntt.h"
using namespace field_config;
using namespace icicle;

eIcicleError cpu_ntt_init_domain(const Device& device, const scalar_t& primitive_root, const NTTInitDomainConfig& config)
{
    auto err = ntt_template::cpu_ntt_init_domain(device, primitive_root, config);
    return err;
}

eIcicleError cpu_ntt_release_domain(const Device& device)
{
    auto err = ntt_template::cpu_ntt_release_domain(device);
    return err;
}

template <typename S, typename E>
eIcicleError cpu_ntt(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
{
    auto err = ntt_template::cpu_ntt<S, E>(device, input, size, dir, config, output);
    return err;
}

eIcicleError cpu_ntt_ref(const Device& device, const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
{
    auto err = ntt_template::cpu_ntt_ref<scalar_t>(device, input, size, dir, config, output);
    return err;
}

REGISTER_NTT_INIT_DOMAIN_BACKEND("CPU", (cpu_ntt_init_domain));
REGISTER_NTT_INIT_DOMAIN_BACKEND("CPU_REF", (cpu_ntt_init_domain));

REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CPU", cpu_ntt_release_domain);
REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CPU_REF", cpu_ntt_release_domain);


REGISTER_NTT_BACKEND("CPU", (cpu_ntt<scalar_t, scalar_t>));
REGISTER_NTT_BACKEND("CPU_REF", (cpu_ntt_ref));

#ifdef EXT_FIELD
REGISTER_NTT_EXT_FIELD_BACKEND("CPU", (cpu_ntt<scalar_t, extension_t>));
#endif // EXT_FIELD