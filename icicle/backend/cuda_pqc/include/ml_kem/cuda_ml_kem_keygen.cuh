#include "ml_kem/pke/cuda_pke_keygen.cuh"
#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_poly.cuh"

namespace icicle::pqc::ml_kem {
  template <const uint8_t k, const uint8_t eta1>
  __forceinline__ __device__ void ml_kem_keygen_internal(
    const uint8_t d[32],
    const uint8_t z[32], // this points to 32 bytes after d. (we allocate d and z together like so: d0, z0, d1, z1, ...
                         // d_{batch - 1}, z_{batch - 1})
    uint8_t ek[384 * k + 32],
    uint8_t dk[768 * k + 96],
    Zq A[256 * k * k])
  {
    pke::keygen<k, eta1>(d, ek, dk, PolyMatrix<256, k, k, Zq>(A));

    // Have wrap 0 (threads 0-31) calculate the last hash
    if (threadIdx.x < 32) {
      H<k>(ek, dk + (384 * k) + (384 * k + 32)); // Point to H(ek) location in dk
    }
    // Have wrap 1 (threads 32-63) copy ek to dk
    else if (threadIdx.x >= 32 && threadIdx.x < 64) {
      // Calculate source and destination indices
      const uint32_t lane = threadIdx.x % 32;
      const uint32_t num_words = (384 * k + 32) / 4;

      // Copy in uint32_t chunks
      for (int i = lane; i < num_words; i += 32) {
        ((uint32_t*)(dk + 384 * k))[i] = ((uint32_t*)ek)[i];
      }
    }
    // copy z to the end of dk with the first 8 threads of wrap2
    else if (threadIdx.x >= 64 && threadIdx.x < 64 + 8) {
      const uint32_t lane = threadIdx.x % 32;
      ((uint32_t*)(dk + 768 * k + 96 - 32))[lane] = ((uint32_t*)z)[lane];
    }
  }
} // namespace icicle::pqc::ml_kem