#pragma once

/********* CPU Backend Configurations *********/

/********* CUDA Backend Configurations *********/

namespace CudaBackendConfig {
  // Enumeration for NTT (Number Theoretic Transform) algorithms
  enum NttAlgorithm {
    Auto,      ///< Automatically choose the algorithm
    Radix2,    ///< Use Radix-2 algorithm
    MixedRadix ///< Use Mixed Radix algorithm
  };

  // Backend-specific configuration flags as constexpr strings
  constexpr const char* CUDA_NTT_FAST_TWIDDLES_MODE = "fast_twiddles"; ///< Enable fast twiddles mode
  constexpr const char* CUDA_NTT_ALGORITHM = "ntt_algorithm";          ///< Set NTT algorithm
} // namespace CudaBackendConfig