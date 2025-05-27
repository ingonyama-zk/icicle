#pragma once

#include <cuda_runtime.h>

/// @file
/// @brief Device-level CUDA code for ML-KEM cryptographic operations.
///
/// This file defines CUDA `__device__` and `__global__` routines for use within
/// batched ML-KEM implementations on the GPU. It can be:
/// - Directly used in fused kernels
/// - Called via the ICICLE runtime, abstracting the backend

namespace icicle {
  namespace pqc {
    namespace ml_kem {

      /// @brief Device placeholder for ML-KEM operations.
      /// Replace this with actual device-level cryptographic logic.
      /// Could be templated later for different parameter sets.
      __device__ void ml_kem_keygen_device_stub()
      {
        // TODO: Replace with actual ML-KEM keygen implementation
        printf("ML-KEM keygen device stub called\n");
      }

      /// @brief CUDA kernel placeholder for batched ML-KEM key generation.
      __global__ void ml_kem_keygen_kernel_stub() { ml_kem_keygen_device_stub(); }

    } // namespace ml_kem
  } // namespace pqc
} // namespace icicle