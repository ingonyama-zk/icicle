#pragma once

#include "ml_kem/ring/cuda_poly.cuh"
#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/hash/cuda_hashes.cuh"
#include "ml_kem/pke/cuda_pke_encrypt.cuh"

namespace icicle::pqc::ml_kem {
  template <const uint8_t k, const uint8_t eta1, const uint8_t eta2, const uint8_t du, const uint8_t dv>
  __forceinline__ __device__ void encaps_internal(
    const uint8_t ek[384 * k + 32],
    const uint8_t m[32],
    const PolyMatrix<256, k, k, Zq> A,
    uint8_t shared_key[32],
    uint8_t c[32 * (du * k + dv)])
  {
    // randomness must remain private, so we cannot expose it
    __shared__ __align__(16) uint8_t k_r[64];
    if (threadIdx.x < 32) { generate_k_r<k>(m, ek, k_r); }
    pke::encrypt<k, eta1, eta2, du, dv>(ek, m, (uint64_t*)(k_r + 32), c, A);
    if (threadIdx.x < 32) { shared_key[threadIdx.x] = k_r[threadIdx.x]; }
  }
} // namespace icicle::pqc::ml_kem