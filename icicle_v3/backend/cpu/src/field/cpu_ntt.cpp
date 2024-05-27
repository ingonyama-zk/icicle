
#include "icicle/ntt/ntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "fields/field_config.h"

using namespace field_config;
using namespace icicle;

template <typename S = scalar_t, typename E = scalar_t>
eIcicleError CpuNtt(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
{
  std::cerr << "CpuNtt() not implemented" << std::endl;
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_NTT_BACKEND("CPU", (CpuNtt<scalar_t, scalar_t>));