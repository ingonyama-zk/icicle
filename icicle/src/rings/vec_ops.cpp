#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"
#include "icicle/rings/random_sampling.h"

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
  eIcicleError ring_random_sampling(
    uint64_t size,
    bool fast_mode,
    const std::byte* seed,
    uint64_t seed_len,
    const VecOpsConfig& config,
    field_t* output)
  {
    return RandomSamplingRingDispatcher::execute(size, fast_mode, seed, seed_len, config, output);
  }

  ICICLE_DISPATCHER_INST(RandomSamplingRingRqDispatcher, ring_rq_random_sampling, ringRqRandomSamplingImpl);
  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, random_sampling_polyring)(
    uint64_t size, bool fast_mode, const std::byte* seed, uint64_t seed_len, const VecOpsConfig* config, Rq* output)
  {
    return RandomSamplingRingRqDispatcher::execute(size, fast_mode, seed, seed_len, *config, output);
  }

  template <>
  eIcicleError ring_random_sampling(
    uint64_t size, bool fast_mode, const std::byte* seed, uint64_t seed_len, const VecOpsConfig& config, Rq* output)
  {
    return RandomSamplingRingRqDispatcher::execute(size, fast_mode, seed, seed_len, config, output);
  }
} // namespace icicle