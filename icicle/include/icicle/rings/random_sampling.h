#pragma once

#include "icicle/errors.h"
#include "icicle/vec_ops.h"

namespace icicle {
  // This should be the same for all the devices to get a deterministic result
  const uint64_t RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS = 256;

  template <typename T>
  eIcicleError ring_random_sampling(
    size_t size, bool fast_mode, const std::byte* seed, size_t seed_len, const VecOpsConfig& config, T* output);
} // namespace icicle