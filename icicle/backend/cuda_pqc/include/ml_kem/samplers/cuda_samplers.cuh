#pragma once

#include "ml_kem/ring/cuda_poly.cuh"
#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/samplers/cuda_sample_helpers.cuh"
#include "ml_kem/samplers/cuda_sample_helpers_5threads.cuh"

#define MAX_HASHES_PER_WRAP 6 // 32 / 5 = 6

namespace icicle::pqc::ml_kem {

  template <const uint8_t k, const uint8_t start_warp = 3, const uint8_t end_warp = 3>
  __forceinline__ __device__ void generate_matrix_A(const uint64_t seed[4], PolyMatrix<256, k, k, Zq> matrix_result)
  {
    const uint8_t warp_idx = threadIdx.x / 32;

    if constexpr (start_warp > 0) { // constexpr to avoid compile-time warnings when start_warp == 0
      if (warp_idx < start_warp || warp_idx > end_warp) return;
    } else {
      if (warp_idx > end_warp) return;
    }

    // ~~~~~~~~~~~ calculating how to divide the work between the warps (done at compile time mostly) ~~~~~~~~~~~
    const uint8_t lane = threadIdx.x % 32;
    const uint8_t hash_lane = lane / 5;

    const uint8_t num_elements = k * k; // number of elements in the matrix
    const uint8_t num_warps = end_warp - start_warp + 1;
    const uint8_t min_hashes_per_warp = num_elements / num_warps;       // important: this is known at compile time.
    const uint8_t hashes_per_warp_remainder = num_elements % num_warps; // 4,
    // how many hashes calculated using the current warp
    const uint8_t warp_hash_count = min_hashes_per_warp + ((warp_idx - start_warp) < hashes_per_warp_remainder);

    // how many iterations of the loop are needed to calculate all the hashes
    // important!!!!!!!!!: this should be the same as (warp_hash_count + MAX_HASHES_PER_WRAP - 1) / MAX_HASHES_PER_WRAP
    const int hash_iter = (min_hashes_per_warp + MAX_HASHES_PER_WRAP - 1) / MAX_HASHES_PER_WRAP;

    const uint8_t start_idx =
      (warp_idx - start_warp) * min_hashes_per_warp + min(warp_idx - start_warp, hashes_per_warp_remainder);
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // We split the loop into two parts:
    // 1. A fully unrollable loop for the minimum number of complete iterations that we know at compile time
    // 2. A final iteration that handles any remaining hashes
#pragma unroll
    for (int iter = 0; iter < hash_iter - 1; iter++) {
      const int l = iter * MAX_HASHES_PER_WRAP + start_idx + hash_lane;
      const uint8_t i = l / k;
      const uint8_t j = l % k;
      sampleNTT5(seed, matrix_result.data() + 256 * l, (i << 8) | j, lane < 30);
    }

    // indicate if the current 5 threaded group inside the warp should calculate a hash
    bool is_active = (((hash_iter - 1) * MAX_HASHES_PER_WRAP + hash_lane) < warp_hash_count) && (lane < 30);
    const int l = ((hash_iter - 1) * MAX_HASHES_PER_WRAP + hash_lane) + start_idx;
    const uint8_t i = l / k;
    const uint8_t j = l % k;
    sampleNTT5(seed, matrix_result.data() + 256 * l, (i << 8) | j, is_active);
  }

  /*
  this function assumes that all of the wraps in the block participate in the matrix generation.
  */
  template <
    const uint8_t k,
    const uint8_t num_elements,
    const uint8_t eta,
    const uint8_t N = 0,
    const bool ntt = false,
    const uint8_t start_warp = 3,
    const uint8_t end_warp = 3>
  __forceinline__ __device__ void
  generate_error_vector(const uint64_t seed[4], PolyVec<256, num_elements, Zq> error_vector_result)
  {
    const uint8_t warp_idx = threadIdx.x / 32;

    if constexpr (start_warp > 0) { // constexpr to avoid compile-time warnings when start_warp == 0
      if (warp_idx < start_warp || warp_idx > end_warp) return;
    } else {
      if (warp_idx > end_warp) return;
    }

    // ~~~~~~~~~~~ calculating how to divide the work between the warps (done at compile time mostly) ~~~~~~~~~~~
    const uint8_t lane = threadIdx.x % 32;
    const uint8_t hash_lane = lane / 5;

    const uint8_t num_warps = end_warp - start_warp + 1;
    const uint8_t min_hashes_per_warp = num_elements / num_warps; // important: this should be known at compile time.
    const uint8_t hashes_per_warp_remainder = num_elements % num_warps; // 4,
    const uint8_t warp_hash_count = min_hashes_per_warp + ((warp_idx - start_warp) < hashes_per_warp_remainder);

    // important!!!!!!!!!: this should be the same as (warp_hash_count + MAX_HASHES_PER_WRAP - 1) / MAX_HASHES_PER_WRAP
    const int hash_iter = (min_hashes_per_warp + MAX_HASHES_PER_WRAP - 1) / MAX_HASHES_PER_WRAP;
    const uint8_t start_idx =
      (warp_idx - start_warp) * min_hashes_per_warp + min(warp_idx - start_warp, hashes_per_warp_remainder);
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // We split the loop into two parts:
    // 1. A fully unrollable loop for the minimum number of complete iterations that we know at compile time
    // 2. A final iteration that handles any remaining hashes
#pragma unroll
    for (int i = 0; i < hash_iter; i++) {
      const uint8_t start_idx_in_warp = i * MAX_HASHES_PER_WRAP + start_idx + hash_lane;
      if constexpr (eta == 3) {
        samplePolyCBD_3_5threads(
          seed, error_vector_result[start_idx_in_warp].data(), start_idx_in_warp + N,
          // the number of hashes to calculate in this warp for this iteration
          min(MAX_HASHES_PER_WRAP, warp_hash_count - i * MAX_HASHES_PER_WRAP));
      } else {
        samplePolyCBD_2_5threads(
          seed, error_vector_result[start_idx_in_warp].data(), start_idx_in_warp + N,
          // the number of hashes to calculate in this warp for this iteration
          min(MAX_HASHES_PER_WRAP, warp_hash_count - i * MAX_HASHES_PER_WRAP));
      }
    }

    if constexpr (ntt) {
#pragma unroll
      for (int t = 0; t < min_hashes_per_warp; t++) {
        ntt_inplace32(error_vector_result[t + start_idx]);
      }

      for (int t = min_hashes_per_warp; t < warp_hash_count; t++) {
        ntt_inplace32(error_vector_result[t + start_idx]);
      }
    }
  }

} // namespace icicle::pqc::ml_kem