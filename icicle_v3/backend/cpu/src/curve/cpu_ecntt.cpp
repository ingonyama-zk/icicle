
#include "icicle/ecntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "ntt_template.h"

#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

template <typename S, typename E>
eIcicleError cpu_ntt(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
{
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_ECNTT_BACKEND("CPU", (cpu_ntt<scalar_t, projective_t>));
