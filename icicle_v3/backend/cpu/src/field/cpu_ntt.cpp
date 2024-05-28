
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