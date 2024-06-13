
#include "icicle/ecntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "ntt.cuh"

#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

template <typename S, typename E>
eIcicleError ecntt_cuda(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
{
  auto err = ntt::ntt_cuda<S, E>(input, size, dir, config, device.id, output);
  return translateCudaError(err);
}

REGISTER_ECNTT_BACKEND("CUDA", (ecntt_cuda<scalar_t, projective_t>));
