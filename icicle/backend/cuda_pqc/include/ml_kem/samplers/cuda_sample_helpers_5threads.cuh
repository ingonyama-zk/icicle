#pragma once

#include "ml_kem/samplers/cuda_sample_utils.cuh"

namespace icicle::pqc::ml_kem {
  __device__ void load_coeffs_shake5_threads(Zq* poly, int& j, const uint8_t& lane)
  {
    uint32_t active = __ballot_sync(MASK, j < 256);
    while (active != 0) {
      uint32_t idx = __ffs(active) - 1;
      Zq* poly_temp = (Zq*)__shfl_sync(MASK, (unsigned long long)poly, idx);
      int j_temp = __shfl_sync(MASK, j, idx);
      const int hash_idx = idx / 5;
      StateSlice slice{static_cast<int>(threadIdx.x / 32), hash_idx};
      load_coeffs_shake(slice.linear_indexing(lane), poly_temp, j_temp, lane);
      j_temp = __shfl_sync(MASK, j_temp, 27); // the 27th thread in the warp holds the highest j value
      j = ((idx / 5) == (lane / 5)) ? j_temp : j;
      active &= ~((1u << (idx + 5)) - 1);
    }
  }

  __device__ void sampleNTT5(const uint64_t rho[4], Zq* poly, uint32_t ij, bool is_active)
  {
    const uint8_t lane = threadIdx.x % 32;
    const uint8_t hash_lane = lane / 5;
    const uint8_t wrap_idx = threadIdx.x / 32;

    StateSlice slice{static_cast<int>(wrap_idx), hash_lane};

    // absorb the input data as shake_128
    if (is_active)
      absorb5_threads<32, SHAKE_128_RATE, (SHAKE_DELIM_BITS << 16), SHAKE_DELIM_SUFFIX>(slice, (const uint8_t*)rho, ij);

    int j = is_active ? 0 : 256;
    while (__any_sync(MASK, j < 256)) {
      if (j < 256) keccakf5_threads(slice);
      load_coeffs_shake5_threads(poly, j, lane);
    }
  }

  __device__ void samplePolyCBD_2_5threads(const uint64_t sigma[4], Zq* poly, uint32_t N, uint8_t num_hashes)
  {
    const uint8_t lane = threadIdx.x % 32;
    const uint8_t hash_lane = lane / 5;
    const uint8_t wrap_idx = threadIdx.x / 32;

    StateSlice slice{static_cast<int>(wrap_idx), hash_lane};

    if (hash_lane < num_hashes) {
      // absorb the input data as shake_256
      absorb5_threads<32, SHAKE_256_RATE, (SHAKE_DELIM_BITS << 8), SHAKE_DELIM_SUFFIX>(slice, (const uint8_t*)sigma, N);
      keccakf5_threads(slice);
    };

    for (int j = 0; j < num_hashes; j++) {
      // get pointer to the current poly
      Zq* poly_temp = (Zq*)__shfl_sync(MASK, (unsigned long long)poly, j * 5);

      StateSlice temp_slice{static_cast<int>(wrap_idx), j};
      uint32_t s_small = temp_slice.linear_indexing_32(lane);
      uint32_t sum = sum_2_bits_and_subtract(s_small);
#pragma unroll
      for (int i = 0; i < 7; i++) {
        poly_temp[8 * lane + i] = Zq::from_raw(sum & 0xF) - 4;
        sum >>= 4;
      }

      poly_temp[8 * lane + 7] = Zq::from_raw(sum & 0xF) - 4;
    }
  }

  __device__ void samplePolyCBD_3_5threads(const uint64_t sigma[4], Zq* poly, uint32_t N, uint8_t num_hashes)
  {
    const uint8_t lane = threadIdx.x % 32;
    const uint8_t lane5 = lane % 5;
    const uint8_t hash_lane = lane / 5;
    const uint8_t wrap_idx = threadIdx.x / 32;

    StateSlice slice{static_cast<int>(wrap_idx), hash_lane};

    if (hash_lane < num_hashes) {
      // absorb the input data as shake_256
      absorb5_threads<32, SHAKE_256_RATE, (SHAKE_DELIM_BITS << 8), SHAKE_DELIM_SUFFIX>(slice, (const uint8_t*)sigma, N);
      keccakf5_threads(slice);
      for (int i = 0; i < 3; i++) {
        ((uint64_t*)poly)[lane5 + i * 5] = slice(i, lane5);
      }
      if (lane5 < 2) ((uint64_t*)poly)[lane5 + 3 * 5] = slice(3, lane5);
      keccakf5_threads(slice);
      ((uint64_t*)poly)[17 + lane5] = slice(0, lane5);
      if (lane5 < 2) ((uint64_t*)poly)[17 + lane5 + 5] = slice(1, lane5);
    };

    if (lane < 24) {
      for (int j = 0; j < num_hashes; j++) {
        Zq* poly_temp = (Zq*)__shfl_sync(0xFFFFFF, (unsigned long long)poly, j * 5);
        uint64_t s = ((uint64_t*)poly_temp)[lane];
        uint32_t s_upper = ((uint32_t*)poly_temp)[(lane + 1) * 2]; // get the lower 32 bits of the next s
        const uint8_t shift = 2 * (lane % 3); // shift is the number of bits that the previous thread processes in s
        s_upper &= (1 << (2 * ((lane + 1) % 3))) - 1;
        uint32_t s_high = (uint32_t)(s >> 32);
        s_upper = __funnelshift_l(s_high, s_upper, 4 - shift);
        uint8_t sum8 = sum_3_bits_and_subtract_8((uint8_t)s_upper);
        uint64_t sum = sum_3_bits_and_subtract(s >> shift);

#pragma unroll
        for (int i = 0; i < 10; i++) {
          poly_temp[lane * 11 - (lane / 3) + i] = Zq::from_raw(sum & 0xF) - 4;
          sum >>= 6;
        }

        if (lane % 3 != 2) { poly_temp[lane * 11 - (lane / 3) + 10] = Zq::from_raw(sum8) - 4; }
      }
    }

    __syncwarp();
  }
} // namespace icicle::pqc::ml_kem