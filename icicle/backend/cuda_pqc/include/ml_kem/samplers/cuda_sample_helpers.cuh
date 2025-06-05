#pragma once

#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/hash/cuda_sha3_32threads.cuh"
#include "ml_kem/samplers/cuda_sample_utils.cuh"

namespace icicle::pqc::ml_kem {
  __device__ void samplePolyCBD_2(const uint64_t sigma[4], Zq* poly, uint32_t N)
  {
    uint64_t s =
      absorb<32, SHAKE_256_RATE, (SHAKE_DELIM_BITS << 8), SHAKE_DELIM_SUFFIX, false>((const uint8_t*)sigma, N);
    s = keccakf(s);
    const uint8_t lane = threadIdx.x % 32;
    uint32_t s_small = __shfl_up_sync(MASK, (uint32_t)(s >> 32), 16);
    s_small = lane < 16 ? ((uint32_t)s) : s_small;
    uint32_t sum = sum_2_bits_and_subtract(s_small);

    const uint8_t lane16 = lane % 16;
// todo: save 2 at a time
#pragma unroll
    for (int i = 0; i < 7; i++) {
      poly[lane16 * 16 + 8 * (lane >= 16) + i] = Zq::from_raw(sum & 0xF) - 4;
      sum >>= 4;
    }

    poly[lane16 * 16 + 8 * (lane >= 16) + 7] = Zq::from_raw(sum & 0xF) - 4;
  }

  __device__ void samplePolyCBD_3(const uint64_t sigma[4], Zq* poly, uint32_t N)
  {
    uint8_t lane = threadIdx.x % 32;
    uint64_t s =
      absorb<32, SHAKE_256_RATE, (SHAKE_DELIM_BITS << 8), SHAKE_DELIM_SUFFIX, false>((const uint8_t*)sigma, N);
    s = keccakf(s);
    if (lane < 17) ((uint64_t*)poly)[lane] = s;
    s = keccakf(s);
    if (lane < 7) ((uint64_t*)poly)[lane + 17] = s;
    __syncwarp();

    if (lane >= 24) return;

    // 30 2 | 1 3 24 3 1 | 2 30 | x 16

    // 60 3 1 | 2 60 2 | 1 3 60 | x 8
    s = ((uint64_t*)poly)[lane];
    uint32_t s_upper = ((uint32_t*)poly)[(lane + 1) * 2]; // get the lower 32 bits of the next s
    const uint8_t shift = 2 * (lane % 3); // shift is the number of bits that the previous thread processes in s
    s_upper &= (1 << (2 * ((lane + 1) % 3))) - 1;
    uint32_t s_high = (uint32_t)(s >> 32);
    s_upper = __funnelshift_l(s_high, s_upper, 4 - shift);
    uint8_t sum8 = sum_3_bits_and_subtract_8((uint8_t)s_upper);
    uint64_t sum = sum_3_bits_and_subtract(s >> shift);

// todo: avoid memory bank conflicts, currently we got -
// 0 5 11 16 21 27 0 5 11 16 21 27 0 5 11 16 21 27 0 5 11 16 21 27
#pragma unroll
    for (int i = 0; i < 10; i++) {
      poly[lane * 11 - (lane / 3) + i] = Zq::from_raw(sum & 0xF) - 4;
      sum >>= 6;
    }

    if (lane % 3 != 2) { poly[lane * 11 - (lane / 3) + 10] = Zq::from_raw(sum8) - 4; }
  }

  __device__ void sampleNTT(const uint64_t rho[4], Zq* poly, uint32_t ij)
  {
    // absorb the input data as shake_128
    uint64_t s =
      absorb<32, SHAKE_128_RATE, (SHAKE_DELIM_BITS << 16), SHAKE_DELIM_SUFFIX, false>((const uint8_t*)rho, ij);

    uint8_t lane = threadIdx.x % 32;
    int j = 0;

    do {
      s = keccakf(s);
      load_coeffs_shake(s, poly, j, lane);
      // Broadcast j value from thread 27 to all threads in warp
      j = __shfl_sync(MASK, j, 27); // the 27th thread in the warp holds the highest j value
    } while (j < 256);
  }
} // namespace icicle::pqc::ml_kem