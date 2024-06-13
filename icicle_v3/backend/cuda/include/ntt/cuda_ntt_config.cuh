#pragma once

#include <stdint.h>

enum class NttAlgorithm : int { Auto, Radix2, MixedRadix };

// backend specific flags
#define CUDA_NTT_FAST_TWIDDLES_MODE "fast_twiddles"
#define CUDA_NTT_ALGORITHM          "ntt_algorithm"
