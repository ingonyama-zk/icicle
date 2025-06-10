#pragma once

#include "ml_kem/ring/cuda_poly.cuh"
#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_zq_math.cuh"
#include "ml_kem/samplers/cuda_samplers.cuh"
#include "ml_kem/hash/cuda_hashes.cuh"
#include "ml_kem/packing/cuda_pack.cuh"

namespace icicle::pqc::ml_kem::pke {

  template <const uint8_t k, const uint8_t eta1>
  __forceinline__ __device__ void
  keygen(const uint8_t d[32], uint8_t ek_pke[384 * k + 32], uint8_t dk_pke[384 * k], PolyMatrixView<256, k, k, Zq> A)
  {
    __shared__ __align__(256) Zq s_e[256 * k * 2];  // s = s_e[0 : 256 * k], e = s_e[256 * k : 256 * k * 2]
    __shared__ __align__(64) uint64_t rho_sigma[8]; // rho = rho_sigma[0 : 4], sigma = rho_sigma[4 : 8]

    // only the first wrap calculate rho and sigma
    if (threadIdx.x < 32) {
      // calculate G(d || k)
      G<k>(d, rho_sigma);
    }

    __syncthreads();

    // Each warp either helps generate matrix A or helps generate the error vector
    switch (k) {
    case 2:
      generate_matrix_A<k, 2, 3>(rho_sigma, A);
      generate_error_vector<k, 2 * k, eta1, 0, true, 1, 1>(rho_sigma + 4, PolyVecView<256, 2 * k, Zq>(s_e));
      break;
    case 3:
      generate_matrix_A<k, 2, 3>(rho_sigma, A);
      generate_error_vector<k, 2 * k, eta1, 0, true, 0, 1>(rho_sigma + 4, PolyVecView<256, 2 * k, Zq>(s_e));
      break;
    case 4:
      generate_matrix_A<k, 1, 3>(rho_sigma, A);
      generate_error_vector<k, 2 * k, eta1, 0, true, 0, 0>(rho_sigma + 4, PolyVecView<256, 2 * k, Zq>(s_e));
      break;
    default:
      __builtin_unreachable();
    }

    // Save rho to the end of ek using the last warp
    if (threadIdx.x >= 96 && threadIdx.x < 104) {
      ((uint32_t*)(ek_pke + 384 * k))[threadIdx.x % 32] = ((uint32_t*)rho_sigma)[threadIdx.x % 32];
    }

    __syncthreads();

    // calculate t_hat = A * s_hat + e
    Zq* t_hat = (Zq*)(s_e + 256 * k);
    PolyVecView<256, k, Zq> vecS(s_e);
    PolyVecView<256, k, Zq> vecT(t_hat);
    matrix_vec_mult<false, true, k>(A, vecS, vecT);

    // save s_hat to global memory (each thread saves 3 bytes)
    for (int i = 0; i < k; i++) {
      byteEncode12(s_e + 256 * i, dk_pke + 384 * i, threadIdx.x);
    }

    // save t_hat to global memory (each thread saves 3 bytes)
    for (int i = 0; i < k; i++) {
      byteEncode12(t_hat + 256 * i, ek_pke + 384 * i, threadIdx.x);
    }

    __syncthreads();
  }
} // namespace icicle::pqc::ml_kem::pke