#pragma once

#include <cuda_runtime.h>
#include "ml-kem/ring/cuda_zq.cuh"
#include "ml-kem/cuda_ml_kem_keygen.cuh"
#include "ml-kem/ring/cuda_poly.cuh"
#include "ml-kem/cuda_ml_kem_encaps.cuh"
#include "ml-kem/cuda_ml_kem_decaps.cuh"

/// @file
/// @brief Device-level CUDA code for ML-KEM cryptographic operations.
///
/// This file defines CUDA `__device__` and `__global__` routines for use within
/// batched ML-KEM implementations on the GPU. It can be:
/// - Directly used in fused kernels
/// - Called via the ICICLE runtime, abstracting the backend

namespace icicle::pqc::ml_kem {

  template <const uint8_t k, const uint8_t eta1>
  __launch_bounds__(128) __global__ void ml_kem_keygen_kernel(
    const uint8_t* entropy, // this points to 32 bytes after d. (we allocate d and z together like so: d0, z0, d1, z1,
                            // ... d_{batch - 1}, z_{batch - 1})
    uint8_t* ek,
    uint8_t* dk,
    Zq* A)
  {
    // update the pointers according to the batch index
    const uint8_t* d = entropy + blockIdx.x * 64; // 32 bytes for d, 32 bytes for z
    const uint8_t* z = d + 32;
    ek += blockIdx.x * (384 * k + 32);
    dk += blockIdx.x * (768 * k + 96); // (dk_pke || ek || H(ek) || z)
    A += blockIdx.x * 256 * k * k;

    ml_kem_keygen_internal<k, eta1>(d, z, ek, dk, A);
  }

  template <const uint8_t k, const uint8_t eta_1, const uint8_t eta_2, const uint8_t du, const uint8_t dv>
  __launch_bounds__(128) __global__
    void ml_kem_encaps_kernel(const uint8_t* ek, const uint8_t* m, uint8_t* K, uint8_t* c, Zq* A)
  {
    ek += blockIdx.x * (384 * k + 32);
    m += blockIdx.x * 32;
    K += blockIdx.x * 32;
    c += blockIdx.x * (32 * (du * k + dv));
    A += blockIdx.x * k * k * 256;
    encaps_internal<k, eta_1, eta_2, du, dv>(ek, m, PolyMatrix<256, k, k, Zq>(A), K, c);
  }

  template <const uint8_t k, const uint8_t eta_1, const uint8_t eta_2, const uint8_t du, const uint8_t dv>
  __launch_bounds__(128) __global__ void ml_kem_decaps_kernel(const uint8_t* dk, const uint8_t* c, uint8_t* K, Zq* A)
  {
    dk += blockIdx.x * (768 * k + 96);
    c += blockIdx.x * (32 * (du * k + dv));
    K += blockIdx.x * 32;
    A += blockIdx.x * k * k * 256;
    decaps_internal<k, eta_1, eta_2, du, dv>(dk, c, PolyMatrix<256, k, k, Zq>(A), K);
  }

} // namespace icicle::pqc::ml_kem