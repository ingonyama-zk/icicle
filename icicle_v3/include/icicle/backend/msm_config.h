#pragma once

/********* CPU Backend Configurations *********/

/********* CUDA Backend Configurations *********/

namespace CudaBackendConfig {
  // Backend-specific configuration flags as constexpr strings
  constexpr const char* CUDA_MSM_IS_BIG_TRIANGLE = "is_big_triangle";
  constexpr const char* CUDA_MSM_LARGE_BUCKET_FACTOR = "large_bucket_factor";
  constexpr const int CUDA_MSM_LARGE_BUCKET_FACTOR_DEFAULT_VAL = 10;
  constexpr const char* CUDA_MSM_NOF_CHUNKS = "nof_chunks";
  constexpr const int CUDA_MSM_NOF_CHUNKS_DEFAULT_VAL = 0;

} // namespace CudaBackendConfig