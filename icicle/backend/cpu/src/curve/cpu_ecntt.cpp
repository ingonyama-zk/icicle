
#include "icicle/backend/ecntt_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "cpu_ntt_main.h"

#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

template <typename S, typename E>
eIcicleError cpu_ntt(const Device& device, const E* input, int size, NTTDir dir, const NTTConfig<S>& config, E* output)
{
  auto err = ntt_cpu::cpu_ntt<S, E>(device, input, size, dir, config, output);
  return err;
}

REGISTER_ECNTT_BACKEND("CPU", (cpu_ntt<scalar_t, projective_t>));
