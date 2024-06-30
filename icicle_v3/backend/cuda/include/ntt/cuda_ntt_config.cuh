#pragma once

#include <stdint.h>

enum class NttAlgorithm : int { Auto, Radix2, MixedRadix };

// backend specific flags
#define CUDA_NTT_FAST_TWIDDLES_MODE "fast_twiddles"
#define CUDA_NTT_ALGORITHM          "ntt_algorithm"

static inline NttAlgorithm get_ntt_alg_from_config(const icicle::ConfigExtension* ext)
{
  // for some reason this does not compile without this small function. WHY ??
  if (ext && ext->has(CUDA_NTT_ALGORITHM)) { return NttAlgorithm(ext->get<int>(CUDA_NTT_ALGORITHM)); }
  return NttAlgorithm::Auto;
}