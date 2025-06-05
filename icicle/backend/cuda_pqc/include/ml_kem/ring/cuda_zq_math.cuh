#pragma once

#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_poly.cuh"

namespace icicle::pqc::ml_kem {
  __constant__ Zq d_zetas[128] = {
    1,    1729, 2580, 3289, 2642, 630,  1897, 848,  1062, 1919, 193,  797,  2786, 3260, 569,  1746, 296,  2447, 1339,
    1476, 3046, 56,   2240, 1333, 1426, 2094, 535,  2882, 2393, 2879, 1974, 821,  289,  331,  3253, 1756, 1197, 2304,
    2277, 2055, 650,  1977, 2513, 632,  2865, 33,   1320, 1915, 2319, 1435, 807,  452,  1438, 2868, 1534, 2402, 2647,
    2617, 1481, 648,  2474, 3110, 1227, 910,  17,   2761, 583,  2649, 1637, 723,  2288, 1100, 1409, 2662, 3281, 233,
    756,  2156, 3015, 3050, 1703, 1651, 2789, 1789, 1847, 952,  1461, 2687, 939,  2308, 2437, 2388, 733,  2337, 268,
    641,  1584, 2298, 2037, 3220, 375,  2549, 2090, 1645, 1063, 319,  2773, 757,  2099, 561,  2466, 2594, 2804, 1092,
    403,  1026, 1143, 2150, 2775, 886,  1722, 1212, 1874, 1029, 2110, 2935, 885,  2154};

  // Precomputed gamma values for Algorithm 11: MultiplyNTTs (Appendix A)
  __constant__ Zq d_gamma[128] = {
    17,   3312, 2761, 568,  583,  2746, 2649, 680,  1637, 1692, 723,  2606, 2288, 1041, 1100, 2229, 1409, 1920, 2662,
    667,  3281, 48,   233,  3096, 756,  2573, 2156, 1173, 3015, 314,  3050, 279,  1703, 1626, 1651, 1678, 2789, 540,
    1789, 1540, 1847, 1482, 952,  2377, 1461, 1868, 2687, 642,  939,  2390, 2308, 1021, 2437, 892,  2388, 941,  733,
    2596, 2337, 992,  268,  3061, 641,  2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109,  375,  2954, 2549, 780,
    2090, 1239, 1645, 1684, 1063, 2266, 319,  3010, 2773, 556,  757,  2572, 2099, 1230, 561,  2768, 2466, 863,  2594,
    735,  2804, 525,  1092, 2237, 403,  2926, 1026, 2303, 1143, 2186, 2150, 1179, 2775, 554,  886,  2443, 1722, 1607,
    1212, 2117, 1874, 1455, 1029, 2300, 2110, 1219, 2935, 394,  885,  2444, 2154, 1175};

  __forceinline__ __device__ void ntt_inplace32(Poly<256, Zq> poly)
  {
    const int lane = threadIdx.x % 32;
    // assume poly is already in the shared memory
    uint8_t k_offset = 1;
#pragma unroll
    for (int i = 0; i < 7; i++) {
      uint8_t shift = 7 - i;
      uint8_t len = 1u << shift;
#pragma unroll
      for (int j = 0; j < 128; j += 64) {
        uint8_t group = (2 * lane + j) >> shift;
        uint8_t idx = (2 * lane + j) & ((1u << shift) - 1);
        uint8_t start = group << (shift + 1);

        Zq zeta = d_zetas[k_offset + group];
        uint32_t U = ((uint32_t*)poly.data())[(start + idx) / 2];
        uint32_t T = ((uint32_t*)poly.data())[(start + idx + len) / 2];

        // Process lower 16 bits
        Zq u = Zq(U & 0xFFFF);
        Zq t = zeta * Zq(T & 0xFFFF);

        U = (U & 0xFFFF0000) | ((u - t).raw());
        T = (T & 0xFFFF0000) | ((u + t).raw());

        // Process upper 16 bits
        u = Zq((U >> 16) & 0xFFFF);
        t = zeta * Zq((T >> 16) & 0xFFFF);

        U = (U & 0x0000FFFF) | ((uint32_t)((u - t).raw()) << 16);
        T = (T & 0x0000FFFF) | ((uint32_t)((u + t).raw()) << 16);

        ((uint32_t*)poly.data())[(start + idx + len) / 2] = U;
        ((uint32_t*)poly.data())[(start + idx) / 2] = T;
      }
      k_offset += (1u << i);
    }
  }

  __forceinline__ __device__ void intt_inplace32(Poly<256, Zq> poly)
  {
    const int lane = threadIdx.x % 32;
    constexpr Zq d_inv_zq = 3303;
    uint8_t k_offset = 127;

    // INTT has the reverse loop order compared to NTT
    for (int i = 0; i < 7; i++) {
      uint8_t shift = i + 1;
      uint8_t len = 1u << shift;

#pragma unroll
      for (int j = 0; j < 128; j += 64) {
        uint8_t group = (2 * lane + j) >> shift;
        uint8_t idx = (2 * lane + j) & ((1u << shift) - 1);
        uint8_t start = group << (shift + 1);

        Zq zeta = d_zetas[k_offset - group];

        uint32_t U = ((uint32_t*)poly.data())[(start + idx) / 2];
        uint32_t T = ((uint32_t*)poly.data())[(start + idx + len) / 2];

        // Lower 16 bits
        Zq t1 = Zq(U & 0xFFFF);
        Zq t2 = Zq(T & 0xFFFF);
        Zq u = t1 + t2;
        Zq t = zeta * (t1 - t2);

        U = (U & 0xFFFF0000) | (u.raw());
        T = (T & 0xFFFF0000) | (t.raw());

        // Upper 16 bits
        t1 = Zq((U >> 16) & 0xFFFF);
        t2 = Zq((T >> 16) & 0xFFFF);
        u = t1 + t2;
        t = zeta * (t1 - t2);

        U = (U & 0x0000FFFF) | (uint32_t(u.raw()) << 16);
        T = (T & 0x0000FFFF) | (uint32_t(t.raw()) << 16);

        ((uint32_t*)poly.data())[(start + idx) / 2] = U;
        ((uint32_t*)poly.data())[(start + idx + len) / 2] = T;
      }

      k_offset -= (1u << (6 - i)); // 6 = 7 - (i + 1)
      __syncthreads();
    }

    // Scale with inverse of N
    uint32_t* p = (uint32_t*)poly.data();
    if (lane < 64) {
      uint32_t val = p[lane];
      Zq lo(val & 0xFFFF);
      Zq hi((val >> 16) & 0xFFFF);

      lo *= d_inv_zq;
      hi *= d_inv_zq;

      p[lane] = (lo.raw()) | (hi.raw() << 16);
    }
  }

  __forceinline__ __device__ void ntt_inplace(Poly<256, Zq> poly)
  { // assume poly is already in the shared memory
    uint8_t k_offset = 1;
#pragma unroll
    for (uint8_t i = 0; i < 7; i++) {
      uint8_t shift = 7 - i;
      uint8_t group = threadIdx.x >> shift;
      uint8_t idx = threadIdx.x & ((1u << shift) - 1);
      uint8_t start = group << (shift + 1);
      uint8_t len = 1u << shift;

      Zq zeta = d_zetas[k_offset + group];

      Zq u = poly[start + idx];
      Zq t = (zeta * poly[start + idx + len]);

      poly[start + idx + len] = u - t;
      poly[start + idx] = (u + t);
      __syncthreads();

      k_offset += (1u << i);
    }
  }

  __forceinline__ __device__ void intt_inplace(Poly<256, Zq> poly)
  {
    // assume poly is already in the shared memory
    constexpr Zq d_inv_zq = 3303;
    uint8_t k_offset = 127;
#pragma unroll
    for (uint8_t i = 0; i < 7; i++) {
      uint8_t shift = i + 1;
      uint8_t len = 1u << shift;
      uint8_t group = threadIdx.x >> shift;
      uint8_t idx = threadIdx.x & ((1u << shift) - 1);
      uint8_t start = group << (shift + 1);

      Zq zeta = d_zetas[k_offset - group];

      Zq t = poly[start + idx];
      poly[start + idx] = (t + poly[start + idx + len]);

      poly[start + idx + len] = zeta * (poly[start + idx + len] - t);
      __syncthreads();

      k_offset -= (1u << (7 - shift));
    }
    poly[threadIdx.x] *= d_inv_zq;
    poly[threadIdx.x + 128] *= d_inv_zq;
  }

  // Algorithm 12: Computes the product of two degree-one polynomials with respect to a quadratic modulus.
  __forceinline__ __device__ void
  base_case_multiply(const Zq& a0, const Zq& a1, const Zq& b0, const Zq& b1, const Zq& gamma, Zq& out0, Zq& out1)
  {
    out0 = a0 * b0 + a1 * b1 * gamma;
    out1 = a0 * b1 + a1 * b0;
  }

  // Algorithm 11: MultiplyNTTs
  __forceinline__ __device__ void ntt_multiply(const Poly<256, Zq>& a, const Poly<256, Zq>& b, Poly<256, Zq>& out)
  {
    base_case_multiply(
      a[threadIdx.x * 2], a[threadIdx.x * 2 + 1], b[threadIdx.x * 2], b[threadIdx.x * 2 + 1], d_gamma[threadIdx.x],
      out[threadIdx.x * 2], out[threadIdx.x * 2 + 1]);
  }

  // Computes matrix-vector multiplication y = Ax or y = A^T x depending on TRANSPOSED flag
  // Can also compute y += Ax or y += A^T x depending on ACCUMULATE flag
  // A is a k x k matrix of polynomials in NTT form
  // x is a vector of k polynomials in NTT form
  // y is the output vector of k polynomials in NTT form
  template <bool TRANSPOSED = false, bool ACCUMULATE = false, uint8_t k>
  __forceinline__ __device__ void
  matrix_vec_mult(const PolyMatrix<256, k, k, Zq> A, const PolyVec<256, k, Zq> x, PolyVec<256, k, Zq> y)
  {
    // #pragma unroll
    for (int i = 0; i < k; ++i) {
      // Initialize or keep output polynomial y[i] based on ACCUMULATE flag
      if constexpr (!ACCUMULATE) {
        y[i][threadIdx.x * 2] = 0;
        y[i][threadIdx.x * 2 + 1] = 0;
        __syncthreads();
      }

#pragma unroll
      for (int j = 0; j < k; ++j) {
        Zq out0, out1;
        // Multiply appropriate matrix element with vector element
        if constexpr (TRANSPOSED) {
          base_case_multiply(
            A(j, i)[threadIdx.x * 2], A(j, i)[threadIdx.x * 2 + 1], x[j][threadIdx.x * 2], x[j][threadIdx.x * 2 + 1],
            d_gamma[threadIdx.x], out0, out1);
        } else {
          base_case_multiply(
            A(i, j)[threadIdx.x * 2], A(i, j)[threadIdx.x * 2 + 1], x[j][threadIdx.x * 2], x[j][threadIdx.x * 2 + 1],
            d_gamma[threadIdx.x], out0, out1);
        }
        // Accumulate result directly into output polynomial
        y[i][threadIdx.x * 2] += out0;
        y[i][threadIdx.x * 2 + 1] += out1;
      }
    }
  }

  template <uint8_t k>
  __forceinline__ __device__ void
  transposed_vec_vec_mult(const PolyVec<256, k, Zq> a, const PolyVec<256, k, Zq> b, Poly<256, Zq> out)
  {
    // Zero out the output polynomial
    out[threadIdx.x * 2] = 0;
    out[threadIdx.x * 2 + 1] = 0;

#pragma unroll
    for (int i = 0; i < k; ++i) {
      Zq out0, out1;
      // Perform base-case multiplication
      base_case_multiply(
        a[i][threadIdx.x * 2], a[i][threadIdx.x * 2 + 1], b[i][threadIdx.x * 2], b[i][threadIdx.x * 2 + 1],
        d_gamma[threadIdx.x], out0, out1);

      // Accumulate results
      out[threadIdx.x * 2] += out0;
      out[threadIdx.x * 2 + 1] += out1;
    }
  }

} // namespace icicle::pqc::ml_kem