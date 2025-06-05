#pragma once

#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/hash/cuda_hash_consts.cuh"

namespace icicle::pqc::ml_kem {
  // Inline device function to perform 64-bit reduction using addition across threads within a warp
  __forceinline__ __device__ uint64_t __reduce_add_sync_64(uint32_t mask, uint64_t value)
  {
    // First reduce the high 32 bits
    uint32_t high_bits = (uint32_t)(value >> 32);
    high_bits = __reduce_add_sync(mask, high_bits);

    // Then reduce the low 32 bits
    uint32_t low_bits = __reduce_add_sync(mask, (uint32_t)value);

    // Combine the results
    return ((uint64_t)high_bits << 32) | low_bits;
  }

  __device__ void load_coeffs_shake(uint64_t s, Zq* poly, int& j, const uint8_t& lane)
  {
    __shared__ uint64_t bytes[SHAKE_128_RATE / 2];
    uint64_t* rate = bytes + (SHAKE_128_RATE / 8) * (threadIdx.x / 32);
    if (lane < SHAKE_128_RATE / 8) rate[lane] = s;

    if (lane >= 28) return;

    const uint64_t MASK28 = 0xFFFFFFF;

    union {
      uint2 coeffs;
      uint64_t coeffs_u64;
    };
    coeffs.x = ((uint32_t*)rate)[lane + (lane / 2)];
    coeffs.y = ((uint32_t*)rate)[lane + (lane / 2) + 1];
    coeffs_u64 >>= 16 * (lane % 2);

    uint4 coeffs_vec;
    coeffs_vec.x = coeffs_u64 & 0xFFF;
    coeffs_vec.y = (coeffs_u64 >> 12) & 0xFFF;
    coeffs_vec.z = (coeffs_u64 >> 24) & 0xFFF;
    coeffs_vec.w = (coeffs_u64 >> 36) & 0xFFF;
    // Count how many values are less than q (valid coefficients)
    uint64_t valid_count =
      (coeffs_vec.x < Zq::q) + (coeffs_vec.y < Zq::q) + (coeffs_vec.z < Zq::q) + (coeffs_vec.w < Zq::q);
    // valid_count *= (lane < 28);
    valid_count <<= (lane / 2) * 4;

    // by taking the or between all the 21 threads, each thread can calculate how many valid coeffs
    // been preduced in the threads before it.
    uint64_t valid_counts = __reduce_add_sync_64(MASK28, valid_count);

    // only keeping the bits that belong to the threads before this one
    valid_counts &= (1ULL << (((lane / 2) + (lane % 2)) * 4)) - 1;
    valid_counts -= valid_count * (lane % 2);

    // sum each 4 bits
    valid_counts = (valid_counts & 0x0F0F0F0F0F0F0F0F) + ((valid_counts >> 4) & 0x0F0F0F0F0F0F0F0F);
    valid_counts = (valid_counts & 0x00FF00FF00FF00FF) + ((valid_counts >> 8) & 0x00FF00FF00FF00FF);
    valid_counts = (valid_counts & 0x0000FFFF0000FFFF) + ((valid_counts >> 16) & 0x0000FFFF0000FFFF);
    valid_counts = (valid_counts & 0x00000000FFFFFFFF) + ((valid_counts >> 32) & 0x00000000FFFFFFFF);

    j += valid_counts;

    // save the coeffs if they are valid
    // Using __stcs for same-warp access
    if (j < 256 && coeffs_vec.x < Zq::q) __stcs((uint16_t*)poly + j, coeffs_vec.x);
    j += coeffs_vec.x < Zq::q;
    if (j < 256 && coeffs_vec.y < Zq::q) __stcs((uint16_t*)poly + j, coeffs_vec.y);
    j += coeffs_vec.y < Zq::q;
    if (j < 256 && coeffs_vec.z < Zq::q) __stcs((uint16_t*)poly + j, coeffs_vec.z);
    j += coeffs_vec.z < Zq::q;
    if (j < 256 && coeffs_vec.w < Zq::q) __stcs((uint16_t*)poly + j, coeffs_vec.w);
    j += coeffs_vec.w < Zq::q;
  }

  __device__ __forceinline__ uint32_t sum_2_bits_and_subtract(uint32_t s)
  {
    constexpr uint32_t mask = 0x55555555U;
    uint32_t sum = (s & mask) + ((s >> 1) & mask);
    return (0x44444444U | (sum & 0x33333333U)) - ((sum >> 2) & 0x33333333U);
  }

  __device__ __forceinline__ uint64_t sum_3_bits_and_subtract(uint64_t s)
  {
    constexpr uint64_t mask = 0x249249249249249ULL;
    uint64_t sum = (s & mask) + ((s >> 1) & mask) + ((s >> 2) & mask);
    return (0x0104104104104104 | (sum & 0x71C71C71C71C71C7)) - ((sum >> 3) & 0x71C71C71C71C71C7);
  }

  __device__ __forceinline__ uint8_t sum_3_bits_and_subtract_8(uint8_t s)
  {
    constexpr uint8_t mask = 0x09; // 0b00001001
    uint8_t sum = (s & mask) + ((s >> 1) & mask) + ((s >> 2) & mask);
    return (0x04 | (sum & 0xC7)) - ((sum >> 3) & 0xC7);
  }
} // namespace icicle::pqc::ml_kem