#pragma once

#include "ml-kem/ring/cuda_zq_math.cuh"
#include "ml-kem/ring/cuda_poly.cuh"
#include "ml-kem/packing/cuda_pack.cuh"
#include "ml-kem/packing/cuda_unpack.cuh"
#include <cuda/barrier>

namespace icicle::pqc::ml_kem::pke {
  template <const uint8_t k, const uint8_t du, const uint8_t dv>
  __forceinline__ __device__ void
  decrypt(const uint8_t dk_pke[384 * k], const uint8_t c[32 * (du * k + dv)], uint8_t m[32])
  {
    __shared__ Zq poly_data[256 * (2 * k + 2)];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // initialize barrier
    if (threadIdx.x == 0) { init(&bar, 128); }

    PolyVec<256, k, Zq> u(poly_data);
    PolyVec<256, k, Zq> s(poly_data + 256 * k);
    Poly<256, Zq> v(poly_data + 256 * k * 2);
    Poly<256, Zq> w(poly_data + 256 * (k * 2 + 1));

#pragma unroll
    for (int i = 0; i < k; ++i) {
      byteDecode12(dk_pke + i * 384, s[i].data(), threadIdx.x);
    }

    const int warp_idx = threadIdx.x / 32;

    if (warp_idx == 3) { byte_decode_decompress<dv>(c + k * 32 * du, v); }

    if (warp_idx < k) { byte_decode_decompress<du>(c + warp_idx * 32 * du, u[warp_idx]); }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < k; ++i) {
      ntt_inplace(u[i]);
    }

    transposed_vec_vec_mult(s, u, w);
    intt_inplace(w);
    if (threadIdx.x < 128) {
      w[threadIdx.x] = v[threadIdx.x] - w[threadIdx.x];
      w[threadIdx.x + 128] = v[threadIdx.x + 128] - w[threadIdx.x + 128];
    }

    if (threadIdx.x < 32) {
      bar.arrive_and_wait();
      encode_message(w, m);
    } else {
      (void)bar.arrive(); // Explicitly ignore the token for non-waiting threads
    }
  }
} // namespace icicle::pqc::ml_kem::pke