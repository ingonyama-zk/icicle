#pragma once

#include "ml-kem/hash/cuda_sha3_32threads.cuh"

namespace icicle::pqc::ml_kem {
  template <const uint8_t k, const uint8_t du, const uint8_t dv>
  __forceinline__ __device__ void J(const uint8_t z[32], const uint8_t c[32 * (du * k + dv)], uint8_t shared_key[32])
  {
    // k = 2 -> size(c) = 32(10 * 2 + 4) = 32 * 24 = 768
    // k = 3 -> size(c) = 32(10 * 3 + 4) = 32 * 34 = 1088
    // k = 4 -> size(c) = 32(10 * 4 + 4) = 32 * 44 = 1408

    // uint64_t s = absorb_dual<64, SHAKE_256_RATE, SHAKE_DELIM_BITS, SHAKE_DELIM_SUFFIX, false>(z, c, 0);
    uint64_t s = absorb_intermediate_dual<SHAKE_256_RATE>(z, c);
    s = keccakf(s);
    c += SHAKE_256_RATE - 32;

    uint64_t s_new;
    for (int i = 32 * (du * k + dv) - (SHAKE_256_RATE - 32); i > SHAKE_256_RATE; i -= SHAKE_256_RATE) {
      s_new = absorb_intermediate<SHAKE_256_RATE>(c);
      s ^= s_new;
      s = keccakf(s);
      c += SHAKE_256_RATE;
    }

    // last aborb
    s_new = absorb<
      ((32 * (du * k + dv) - (SHAKE_256_RATE - 32)) % SHAKE_256_RATE), SHAKE_256_RATE, SHAKE_DELIM_BITS,
      SHAKE_DELIM_SUFFIX, false>(c, 0);
    s ^= s_new;
    s = keccakf(s);

    uint8_t lane = threadIdx.x % 32;
    if (lane < 4) ((uint64_t*)shared_key)[lane] = s;
  }

  __forceinline__ __device__ void G_m_ek(const uint8_t m[32], const uint8_t hashed_ek[32], uint8_t k_r[64])
  {
    uint64_t s = absorb_dual<64, SHA3_512_RATE, SHA3_DELIM_BITS, SHA3_DELIM_SUFFIX>(m, hashed_ek, 0);
    s = keccakf(s);

    uint8_t lane = threadIdx.x % 32;
    if (lane < 8) ((uint64_t*)k_r)[lane] = s;
  }

  template <const uint8_t k>
  __forceinline__ __device__ void H(const uint8_t ek[384 * k + 32], uint8_t dk[32])
  {
    // todo: add a fence here to check that we finished
    // todo: use other wraps to absorb to shared memory and only xor with it when needed

    uint64_t s = absorb_intermediate<SHA3_256_RATE>(ek);
    s = keccakf(s);
    ek += SHA3_256_RATE;

    uint64_t s_new;
    for (int i = (384 * k + 32) - SHA3_256_RATE; i > SHA3_256_RATE; i -= SHA3_256_RATE) {
      s_new = absorb_intermediate<SHA3_256_RATE>(ek);
      s ^= s_new;
      s = keccakf(s);
      ek += SHA3_256_RATE;
    }

    // last aborb
    s_new = absorb<((384 * k + 32) % SHA3_256_RATE), SHA3_256_RATE, SHA3_DELIM_BITS, SHA3_DELIM_SUFFIX, false>(ek, 0);
    s ^= s_new;
    s = keccakf(s);

    // save to global memory
    const uint8_t lane = threadIdx.x % 32;
    if (lane < 4) ((uint64_t*)dk)[lane] = s;
  }

  template <const uint8_t k>
  __forceinline__ __device__ void generate_k_r(const uint8_t m[32], const uint8_t ek[384 * k + 32], uint8_t k_r[64])
  {
    __shared__ __align__(16) uint8_t hashed_ek[32];
    H<k>(ek, hashed_ek);
    G_m_ek(m, hashed_ek, k_r);
  }

  template <uint32_t k>
  __forceinline__ __device__ void G(const uint8_t*& d, uint64_t rho_sigma[8])
  {
    uint64_t s = absorb<32, SHA3_512_RATE, (((uint32_t)SHA3_DELIM_BITS) << 8), SHA3_DELIM_SUFFIX>(d, k);
    s = keccakf(s);

    uint8_t lane = threadIdx.x % 32;
    if (lane < 8) rho_sigma[lane] = s;
  }
}