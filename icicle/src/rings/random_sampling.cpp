#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"
#include "icicle/random_sampling.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(RandomSamplingRingDispatcher, ring_zq_random_sampling, ringZqRandomSamplingImpl);
  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, random_sampling)(
    uint64_t size,
    bool fast_mode,
    const std::byte* seed,
    uint64_t seed_len,
    const VecOpsConfig* config,
    field_t* output)
  {
    return RandomSamplingRingDispatcher::execute(size, fast_mode, seed, seed_len, *config, output);
  }

  template <>
  eIcicleError random_sampling(
    uint64_t size,
    bool fast_mode,
    const std::byte* seed,
    uint64_t seed_len,
    const VecOpsConfig& config,
    field_t* output)
  {
    return RandomSamplingRingDispatcher::execute(size, fast_mode, seed, seed_len, config, output);
  }
} // namespace icicle