#include "icicle/ntt.h"
#include "icicle/dispatcher.h"

namespace icicle {

  // NTT
  ICICLE_DISPATCHER_INST(NttDispatcher, ntt, NttImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(FIELD, ntt)(const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return NttDispatcher::execute(input, size, dir, config, output);
  }

  template <>
  eIcicleError ntt(const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, ntt)(input, size, dir, config, output);
  }

  // INIT DOMAIN
  ICICLE_DISPATCHER_INST(NttInitDomainDispatcher, ntt_init_domain, NttInitDomainImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(FIELD, ntt_init_domain)(const scalar_t& primitive_root, const ConfigExtension& config)
  {
    return NttInitDomainDispatcher::execute(primitive_root, config);
  }

  template <>
  eIcicleError ntt_init_domain(const scalar_t& primitive_root, const ConfigExtension& config)
  {
    return CONCAT_EXPAND(FIELD, ntt_init_domain)(primitive_root, config);
  }

} // namespace icicle