#pragma once

#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_poly.cuh"
#include "ml_kem/cuda_ml_kem_encaps.cuh"
#include "ml_kem/pke/cuda_pke_decrypt.cuh"
#include "ml_kem/pke/cuda_pke_encrypt.cuh"

namespace icicle::pqc::ml_kem {

  namespace {
    template <const uint8_t k, const uint8_t du, const uint8_t dv>
    __forceinline__ __device__ void
    compare_ciphertexts(const uint8_t* __restrict__ c_pke, const uint8_t* __restrict__ c, uint8_t* success)
    {
      constexpr int bytes_amount = 32 * (du * k + dv);
      const int tid = threadIdx.x;

      constexpr int chunk_size = sizeof(uint64_t); // 8 bytes
      constexpr int chunks_amount = bytes_amount / chunk_size;

      auto c_pke64 = reinterpret_cast<const uint64_t*>(c_pke);
      auto c64 = reinterpret_cast<const uint64_t*>(c);

      for (int i = tid; i < chunks_amount; i += blockDim.x) {
        if (c_pke64[i] != c64[i]) {
          *success = 0;
          // printf("Thread %d in block %d entered here\n", threadIdx.x, blockIdx.x);
        }
      }
    }
  } // namespace
  template <const uint8_t k, const uint8_t eta1, const uint8_t eta2, const uint8_t du, const uint8_t dv>
  __forceinline__ __device__ void decaps_internal(
    const uint8_t dk[768 * k + 96],
    const uint8_t c[32 * (du * k + dv)],
    PolyMatrix<256, k, k, Zq> A,
    uint8_t shared_key[32])
  {
    __shared__ uint8_t success;
    if (threadIdx.x == 0) { success = 0xFF; }

    const uint8_t* dk_pke = dk;
    const uint8_t* ek_pke = dk + 384 * k;
    const uint8_t* h = dk + 768 * k + 32;
    const uint8_t* z = dk + 768 * k + 64;

    __shared__ __align__(16) uint8_t m[32];
    pke::decrypt<k, du, dv>(dk_pke, c, m);

    __shared__ __align__(16) uint8_t k_r[64];
    __shared__ __align__(16) uint8_t shared_key_bar[32];
    if (threadIdx.x < 32) {
      G_m_ek(m, h, k_r);
    } else if (threadIdx.x >= 96 && threadIdx.x < 128) {
      J<k, du, dv>(z, c, shared_key_bar);
    }

    __shared__ __align__(16) uint8_t c_pke[32 * (du * k + dv)];
    pke::encrypt<k, eta1, eta2, du, dv>(ek_pke, m, (uint64_t*)(k_r + 32), c_pke, A);

    __syncthreads();

    compare_ciphertexts<k, du, dv>(c_pke, c, &success);
    // __shared__ __align__(16) uint8_t shared_key_bar[32];
    if (threadIdx.x < 32) {
      // todo: make sure compare ciphertexts finished before using succsess in here
      shared_key[threadIdx.x] =
        shared_key_bar[threadIdx.x] ^ (success & (shared_key_bar[threadIdx.x] ^ k_r[threadIdx.x]));
    }
  }
} // namespace icicle::pqc::ml_kem