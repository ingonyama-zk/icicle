#pragma once

#include "ml_kem/samplers/cuda_samplers.cuh"
#include "ml_kem/ring/cuda_zq_math.cuh"
#include "ml_kem/ring/cuda_poly.cuh"
#include "ml_kem/packing/cuda_pack.cuh"
#include "ml_kem/packing/cuda_unpack.cuh"

namespace icicle::pqc::ml_kem::pke {
  template <const uint8_t k, const uint8_t eta1, const uint8_t eta2, const uint8_t du, const uint8_t dv>
  __forceinline__ __device__ void encrypt_with_matrix_A(
    const __restrict__ uint8_t ek_pke[384 * k + 32],
    const __restrict__ uint8_t m[32],
    const __restrict__ uint64_t r[4],
    uint8_t c[32 * (du * k + dv)],
    const PolyMatrixView<256, k, k, Zq>& A)
  {
    __shared__ __align__(16) Zq t_data[256 * k];
    __shared__ __align__(16) Zq u_data[256 * k];
    __shared__ __align__(16) Zq y_data[256 * k];
    __shared__ __align__(16) Zq e1_e2_data[256 * (k + 1)];
    __shared__ __align__(16) Zq mu_data[256];
    __shared__ __align__(16) Zq v_data[256];

    PolyVecView<256, k, Zq> t(t_data);
    PolyVecView<256, k, Zq> u(u_data);
    PolyVecView<256, k, Zq> y(y_data);
    PolyVecView<256, k + 1, Zq> e1_e2(e1_e2_data);
    PolyVecView<256, k, Zq> e1(e1_e2_data);
    PolyView<256, Zq> e2(e1_e2_data + 256 * k);
    generate_error_vector<k, k, eta1, 0, true, 0, 0>(r, y);
    generate_error_vector<k, k + 1, eta2, k, false, 0, 0>(r, e1_e2);

// (1) run ByteDecode_12 𝑘 times to decode 𝐭 ∈ (ℤ_q^256)^𝑘
#pragma unroll
    for (int i = 0; i < k; ++i) {
      byteDecode12(ek_pke + i * 384, t[i].data(), threadIdx.x);
    }

    __syncthreads();

    // 19. u = intt((A^T)*y) + e1
    // 19.a u = (A^T)*y
    matrix_vec_mult<true, false, k>(A, y, u);
    // __syncthreads();

    // 19.b u = intt(u)
#pragma unroll
    for (int i = 0; i < k; ++i) {
      intt_inplace(u[i]);
    }

// 19.c u = u + e1
#pragma unroll
    for (int i = 0; i < k; ++i) {
      u[i][threadIdx.x] += e1[i][threadIdx.x];
      u[i][threadIdx.x + 128] += e1[i][threadIdx.x + 128];
    }

    // 20. unpack message
    // https://github.com/pq-crystals/kyber/blob/main/ref/poly.c#L168-L182
    // __shared__ Zq mu_data[256];
    PolyView<256, Zq> mu(mu_data);
    decode_message(m, mu);

    // 21. intt(t^T ∘ y) + e2 + mu
    // __shared__ Zq v_data[256];
    PolyView<256, Zq> v(v_data);
    // 21.a v = t^T ∘ y
    // returns Zq element in v
    transposed_vec_vec_mult(t, y, v);

    // 21.b v = intt(v)
    intt_inplace(v);
    // 21.c v = v + e2 + mu
    v[threadIdx.x] += (e2[threadIdx.x] + mu[threadIdx.x]);
    v[threadIdx.x + 128] += (e2[threadIdx.x + 128] + mu[threadIdx.x + 128]);

    __syncthreads();

    // 22. 23. pack
    encode_ciphertext<k, du, dv>(u, v, c);
  }

  template <
    const uint8_t k,
    const uint8_t eta1,
    const uint8_t eta2,
    const uint8_t du,
    const uint8_t dv,
    const bool dynamic_A = false>
  __forceinline__ __device__ void encrypt(
    const uint8_t ek_pke[384 * k + 32],
    const uint8_t m[32],
    const uint64_t r[4],
    uint8_t c[32 * (du * k + dv)],
    PolyMatrixView<256, k, k, Zq> A)
  {
    const uint64_t* rou = (const uint64_t*)(ek_pke + 384 * k);
    if constexpr (dynamic_A) {
      switch (k) {
      case 2:
      case 3:
        generate_matrix_A<k, 1, 2>(rou, A);
        break;
      case 4:
        generate_matrix_A<k, 1, 3>(rou, A);
        break;
      default:
        __builtin_unreachable();
      }
    } else {
      generate_matrix_A<k, 1, 2>(rou, A);
    }

    encrypt_with_matrix_A<k, eta1, eta2, du, dv>(ek_pke, m, r, c, A);
  }
} // namespace icicle::pqc::ml_kem::pke