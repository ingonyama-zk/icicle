#pragma once

#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_poly.cuh"

namespace icicle::pqc::ml_kem {
  template <const uint8_t d>
  __forceinline__ __device__ uint16_t compress_element(const Zq in)
  {
    static_assert(d > 0 && d < 12, "d must be in [1, 11]");
    constexpr uint16_t mask = (1u << d) - 1;
    constexpr uint16_t half_q = Zq::q >> 1;
    uint32_t tmp = (uint32_t)in.raw() << d;
    return ((tmp + half_q) / Zq::q) & mask;
  }

  template <const uint8_t d>
  __forceinline__ __device__ void
  byte_encode_compress(const Poly<256, Zq>& input /*shared*/, uint8_t* output_packed /*shared*/)
  {
    // We have n=256 coefficients in Rq and Tq, each is packed from 32 bits to d bits
    // Therefore we choose to use a single warp (32 threads) to encode 8 coefficients per thread
    // Each thread reads 32 bytes and writes 8d Bytes. Note that this is not coalesced really well. Idea?
    static_assert(d > 0 && d <= 12, "d must be in [1, 12]");
    constexpr int num_coeffs = 256;      // this is n
    constexpr int coeffs_per_thread = 8; // Do not modify, the implementation assumes this!
    static_assert(num_coeffs == coeffs_per_thread * 32, "Only 32-thread warp supported");
    // todo: remove this somehow
    if (threadIdx.x >= 32) { return; }

    // Pack into bytes. we need 8*d bits which is at most 8*12=96 bits
    __uint128_t packed_coeffs = 0;
#pragma unroll
    for (int i = 0; i < coeffs_per_thread; ++i) {
      if constexpr (d < 12) {
        packed_coeffs |= (__uint128_t)compress_element<d>(input[threadIdx.x * coeffs_per_thread + i])
                         << (i * d); // shift and set packed bits
      } else {
        packed_coeffs |= (__uint128_t)input[threadIdx.x * coeffs_per_thread + i].raw()
                         << (i * d); // shift and set packed bits
      }
    }

    // Write back packed bytes to output
    // Note that we write d bytes, because we compute 8 coefficients.
    memcpy(output_packed + threadIdx.x * d, &packed_coeffs, d);
  }

  __forceinline__ __device__ void byteEncode12(Zq* input, uint8_t* output, uint32_t index)
  {
    // Load two 12-bit values and pack them into 3 bytes
    const uint16_t val1 = input[index * 2].raw();
    const uint16_t val2 = input[index * 2 + 1].raw();

    output[index * 3] = val1 & 0xff;
    output[index * 3 + 1] = ((val1 >> 8) & 0x0f) | ((val2 & 0x0f) << 4);
    output[index * 3 + 2] = val2 >> 4;
  }

  __forceinline__ __device__ void poly_encode_compress4(const Poly<256, Zq>& a, uint8_t* r)
  {
    if (threadIdx.x >= 32) return;

    const int blk = threadIdx.x; // which block of 8
    uint8_t t[8];
    uint32_t d0;
    uint16_t u_raw;

#pragma unroll
    for (int j = 0; j < 8; j++) {
      u_raw = a[blk * 8 + j].raw();
      d0 = (uint32_t)u_raw << 4;
      d0 += 1665U;
      d0 *= 80635U;
      d0 >>= 28;
      t[j] = d0 & 0x0F;
    }

    const int ro = blk * 4;
    r[ro + 0] = t[0] | (t[1] << 4);
    r[ro + 1] = t[2] | (t[3] << 4);
    r[ro + 2] = t[4] | (t[5] << 4);
    r[ro + 3] = t[6] | (t[7] << 4);
  }

  __forceinline__ __device__ void poly_encode_compress5(const Poly<256, Zq>& a, uint8_t* r)
  {
    if (threadIdx.x >= 32) return;

    const int blk = threadIdx.x; // which block of 8
    uint8_t t[8];
    uint32_t d0;
    uint16_t u_raw;

#pragma unroll
    for (int j = 0; j < 8; j++) {
      u_raw = a[blk * 8 + j].raw();
      d0 = (uint32_t)u_raw << 5;
      d0 += 1664U;
      d0 *= 40318U;
      d0 >>= 27;
      t[j] = d0 & 0x1F;
    }
    const int ro = blk * 5;
    r[ro + 0] = (t[0]) | (t[1] << 5);
    r[ro + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7);
    r[ro + 2] = (t[3] >> 1) | (t[4] << 4);
    r[ro + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6);
    r[ro + 4] = (t[6] >> 2) | (t[7] << 3);
  }

  template <uint8_t k>
  __forceinline__ __device__ void polyvec_encode_compress10(const PolyVec<256, k, Zq>& a, uint8_t* r)
  {
    constexpr int tasks = k * (256 >> 2); // k * 64
    uint16_t t[4];
    uint64_t d0;
    for (int tid = threadIdx.x; tid < tasks; tid += 128) {
      const int i = tid >> 6; // /64
      const int j = tid & 63; // %64
#pragma unroll
      for (int l = 0; l < 4; l++) {
        d0 = (uint64_t)a[i][4 * j + l].raw() << 10;
        d0 += 1665ULL;
        d0 *= 1290167ULL;
        d0 >>= 32;
        t[l] = (uint16_t)(d0 & 0x03FF);
      }

      int r_offset = i * 320 + j * 5;
      r[r_offset + 0] = t[0];
      r[r_offset + 1] = (t[0] >> 8) | (t[1] << 2);
      r[r_offset + 2] = (t[1] >> 6) | (t[2] << 4);
      r[r_offset + 3] = (t[2] >> 4) | (t[3] << 6);
      r[r_offset + 4] = (t[3] >> 2);
    }
  }

  template <uint8_t k>
  __forceinline__ __device__ void polyvec_encode_compress11(const PolyVec<256, k, Zq>& a, uint8_t* r)
  {
    constexpr int total_threads = k * (256 >> 3);
    if (threadIdx.x >= total_threads) return; // k = {2, 3, 4} total_threads = {64, 96, 128}

    int i = threadIdx.x >> 5;
    int j = threadIdx.x & 31;
    uint16_t t[8];
    uint64_t d0;
#pragma unroll
    // TODO: switch to mad operations
    for (int l = 0; l < 8; l++) {
      d0 = a[i][8 * j + l].raw();
      d0 <<= 11;
      d0 += 1664;
      d0 *= 645084;
      d0 >>= 31;
      t[l] = d0 & 0x7ff;
    }

    int r_offset = i * 352 + j * 11;

    r[r_offset + 0] = (t[0] >> 0);
    r[r_offset + 1] = (t[0] >> 8) | (t[1] << 3);
    r[r_offset + 2] = (t[1] >> 5) | (t[2] << 6);
    r[r_offset + 3] = (t[2] >> 2);
    r[r_offset + 4] = (t[2] >> 10) | (t[3] << 1);
    r[r_offset + 5] = (t[3] >> 7) | (t[4] << 4);
    r[r_offset + 6] = (t[4] >> 4) | (t[5] << 7);
    r[r_offset + 7] = (t[5] >> 1);
    r[r_offset + 8] = (t[5] >> 9) | (t[6] << 2);
    r[r_offset + 9] = (t[6] >> 6) | (t[7] << 5);
    r[r_offset + 10] = (t[7] >> 3);
  }

  template <const uint8_t k, const uint8_t du, const uint8_t dv>
  __forceinline__ __device__ void encode_ciphertext(const PolyVec<256, k, Zq>& u, const Poly<256, Zq>& v, uint8_t* c)
  {
    if constexpr (du == 11) {
      polyvec_encode_compress11<k>(u, c);
    } else if constexpr (du == 10) {
      polyvec_encode_compress10<k>(u, c);
    } else {
#pragma unroll
      for (int i = 0; i < k; ++i) {
        byte_encode_compress<du>(u[i], c + i * 32 * du);
      }
    }
    if constexpr (du == 4) {
      poly_encode_compress4(v, c + k * 32 * du);
    } else if constexpr (du == 5) {
      poly_encode_compress5(v, c + k * 32 * du);
    } else {
      byte_encode_compress<dv>(v, c + k * 32 * du);
    }
  }

  __forceinline__ __device__ void encode_message(const Poly<256, Zq>& mu, uint8_t* m)
  {
    uint32_t t;
    m[threadIdx.x] = 0;

#pragma unroll
    for (int j = 0; j < 8; j++) {
      // TODO: switch to mad operations
      t = mu[8 * threadIdx.x + j].raw();
      t <<= 1;
      t += 1665;
      t *= 80635;
      t >>= 28;
      t &= 1;
      m[threadIdx.x] |= t << j;
    }
  }
} // namespace icicle::pqc::ml_kem