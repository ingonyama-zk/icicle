#pragma once

#include "icicle/errors.h"
#include "icicle/vec_ops.h"

namespace icicle {
    template <typename T>
    eIcicleError ring_random_sampling(size_t size, bool fast_mode, const std::byte* seed, size_t seed_len, const VecOpsConfig& config, T* output);
}