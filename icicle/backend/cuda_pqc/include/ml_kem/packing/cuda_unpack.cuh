#pragma once

#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_poly.cuh"

namespace icicle::pqc::ml_kem {
  template <const uint8_t d>
  __forceinline__ __device__ void
  byte_decode_decompress(const uint8_t* input_packed /*shared*/, PolyView<256, Zq> output /*shared*/)
  {
    constexpr int num_coeffs = 256;
    constexpr int coeffs_per_thread = 8;
    constexpr uint16_t mask = ((1u << d) - 1);
    static_assert(d > 0 && d <= 12, "d must be in [1, 12]");
    static_assert(num_coeffs == coeffs_per_thread * 32, "Only 32-thread warp supported");
    constexpr uint16_t half = 1u << (d - 1);

    // Load the d bytes of packed data into a 128‑bit integer
    __uint128_t packed = 0;
    const int lane = threadIdx.x % 32;
    memcpy(&packed, input_packed + lane * d, d);

// Extract the 8 coefficients
#pragma unroll
    for (int i = 0; i < coeffs_per_thread; ++i) {
      if constexpr (d < 12) {
        output[lane * coeffs_per_thread + i] =
          Zq{((uint16_t)((packed >> (i * d)) & mask) * (uint32_t)Zq::q + half) >> d};
      } else {
        output[lane * coeffs_per_thread + i] = Zq{(uint16_t)((packed >> (i * d)) & mask)};
      }
    }
  }

  __forceinline__ __device__ void byteDecode12(const uint8_t* input, Zq* output, uint32_t index)
  {
    // Read 3 bytes and reconstruct two 12-bit values
    const uint8_t b0 = input[index * 3];
    const uint8_t b1 = input[index * 3 + 1];
    const uint8_t b2 = input[index * 3 + 2];

    // First 12-bit value: lower 8 bits from b0, upper 4 bits from b1
    const uint16_t val1 = b0 | ((b1 & 0x0F) << 8);

    // Second 12-bit value: lower 4 bits from b1 (shifted right), upper 8 bits from b2
    const uint16_t val2 = ((b1 >> 4) & 0x0F) | (b2 << 4);

    output[index * 2] = Zq::from_raw(val1);
    output[index * 2 + 1] = Zq::from_raw(val2);
  }

  __forceinline__ __device__ void decode_message(const uint8_t* __restrict__ m, PolyView<256, Zq> mu)
  {
    constexpr uint16_t half_q = 0x681;

    int tid = threadIdx.x;
    int i0 = tid << 1;
    int i1 = i0 + 1;

    int b0 = i0 >> 3, s0 = i0 & 7, s1 = s0 + 1; // b0 == b1

    // Extract bit, sign‐extend into mask: mask = b ? 0xFFFF : 0x0000
    uint16_t mask0 = -uint16_t((m[b0] >> s0) & 1u);
    uint16_t mask1 = -uint16_t((m[b0] >> s1) & 1u);

    // Apply mask to half_q
    mu[i0] = mask0 & half_q;
    mu[i1] = mask1 & half_q;
  }
} // namespace icicle::pqc::ml_kem