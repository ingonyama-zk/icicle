#pragma once

#include "ml_kem/hash/cuda_hash_consts.cuh"

namespace icicle::pqc::ml_kem {
  /*
  the state of each warp looks like this:
  [hash0 state]    [hash1 state]    [hash2 state]    [hash3 state]    [hash4 state]    [hash5 state]
  0 1 2 3 4      | 0 1 2 3 4      | 0 1 2 3 4      | 0 1 2 3 4      | 0 1 2 3 4      | 0 1 2 3 4      | = 30 elements
  5 6 7 8 9      | 5 6 7 8 9      | 5 6 7 8 9      | 5 6 7 8 9      | 5 6 7 8 9      | 5 6 7 8 9      |
  10 11 12 13 14 | 10 11 12 13 14 | 10 11 12 13 14 | 10 11 12 13 14 | 10 11 12 13 14 | 10 11 12 13 14 |
  15 16 17 18 19 | 15 16 17 18 19 | 15 16 17 18 19 | 15 16 17 18 19 | 15 16 17 18 19 | 15 16 17 18 19 |
  20 21 22 23 24 | 20 21 22 23 24 | 20 21 22 23 24 | 20 21 22 23 24 | 20 21 22 23 24 | 20 21 22 23 24 |

  the shape is 4x5x6x5 [4 warps][5 rows][6 hashes in warp][5 columns]
  */
  __shared__ __align__(64) uint64_t sha3_state_raw[BEST_STATE_SIZE * 4 * MAX_HASHES_PER_WRAP];

  // Slice view that handles state access with fixed warp and hash indices
  struct StateSlice {
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_ROWS = 5;
    static constexpr int NUM_HASHES = MAX_HASHES_PER_WRAP;
    static constexpr int NUM_COLS = 5;

    const int warp_idx;
    const int hash_idx;

    // Access as if dimensions were [row][col]
    __device__ __forceinline__ uint64_t& operator()(int row, int col)
    {
      uint64_t* aligned_data = (uint64_t*)__builtin_assume_aligned(sha3_state_raw, 64);
      return aligned_data[((warp_idx * NUM_ROWS + row) * NUM_HASHES + hash_idx) * NUM_COLS + col];
    }

    __device__ __forceinline__ const uint64_t& operator()(int row, int col) const
    {
      const uint64_t* aligned_data = (const uint64_t*)__builtin_assume_aligned(sha3_state_raw, 64);
      return aligned_data[((warp_idx * NUM_ROWS + row) * NUM_HASHES + hash_idx) * NUM_COLS + col];
    }

    // Linear indexing as if dimensions were [row][col]
    __device__ __forceinline__ void linear_indexing(int idx, uint64_t value)
    {
      const int row = idx / 5;
      const int col = idx % 5;
      (*this)(row, col) = value;
    }

    __device__ __forceinline__ uint64_t linear_indexing(int idx) const
    {
      const int row = idx / 5;
      const int col = idx % 5;
      return (*this)(row, col);
    }

    // Get reference to state value for casting
    __device__ __forceinline__ uint64_t& get_state_ref(int idx)
    {
      const int row = idx / 5;
      const int col = idx % 5;
      return (*this)(row, col);
    }

    // Linear indexing for uint32_t access
    __device__ __forceinline__ uint32_t& get_state_ref_32(int idx)
    {
      const int row = (idx / 2) / 5;
      const int col = (idx / 2) % 5;
      return ((uint32_t*)&(*this)(row, col))[idx % 2];
    }

    __device__ __forceinline__ uint32_t linear_indexing_32(int idx) const
    {
      const int row = (idx / 2) / 5;
      const int col = (idx / 2) % 5;
      return ((uint32_t*)&(*this)(row, col))[idx % 2];
    }

    __device__ __forceinline__ void linear_indexing_32(int idx, uint32_t value)
    {
      const int row = (idx / 2) / 5;
      const int col = (idx / 2) % 5;
      ((uint32_t*)&(*this)(row, col))[idx % 2] = value;
    }
  };

  template <const int INPUT_LEN = 32, const int RATE, const uint32_t PAD_DELIM, const uint64_t PAD_SUFFIX>
  __device__ __forceinline__ void absorb5_threads(StateSlice state, const uint8_t d[INPUT_LEN], uint32_t extra_input)
  {
    const int lane = threadIdx.x % 32;
    const int lane5 = lane % 5;

#pragma unroll
    for (int i = 0; i < 5; i++) {
      state(i, lane5) = (lane5 + i * 5 < (INPUT_LEN / 8)) ? ((const uint64_t*)d)[lane5 + i * 5] : 0;
    }

    // add the padding delimiter
    if (lane5 == 0) {
      // add the padding delimiter
      state.linear_indexing(INPUT_LEN / 8, PAD_DELIM | extra_input);
      // add the padding suffix
      state.linear_indexing(RATE / 8 - 1, PAD_SUFFIX);
    }
  }

  __device__ void keccakf5_threads(StateSlice state)
  {
    const int lane = threadIdx.x % 32;
    const int lane5 = lane % 5;
    const int next_col = (lane5 + 1) % 5;
    const int prev_col = (lane5 + 4) % 5;
    const int lane5_inv = (lane5 * 2) % 5; // 0, 2, 4, 1, 3

    for (int j = 0; j < 24; j++) {
      uint64_t ss[5];
      // theta
      ss[0] = state(0, lane5);
      uint64_t c = ss[0];
#pragma unroll
      for (int i = 1; i < 5; i++) {
        ss[i] = state(i, lane5);
        c ^= ss[i];
      }

      state(0, lane5) = c;

      uint64_t c_plus_1 = state(0, next_col);
      c_plus_1 = ROTL1(c_plus_1);
      uint64_t c_minus_1 = state(0, prev_col);
      uint64_t d = c_minus_1 ^ c_plus_1;

      ss[0] ^= d;
      ss[0] = ROTL64(ss[0], rot_amount[lane5]);

#pragma unroll
      for (int i = 1; i < 5; i++) {
        ss[i] ^= d;
        ss[i] = ROTL64(ss[i], rot_amount[lane5 + 5 * i]);
        state(i, lane5) = ss[i];
        int idx = (lane5 + i) % 5; // 0, 6, 12, 18, 24
        ss[i] = state(i, idx);
      }

      // #pragma unroll
      // for (int i = 1; i < 5; i++) {
      //     state(i, lane5) ^= d;
      //     int idx = (lane5 + i) % 5; // 0, 6, 12, 18, 24
      //     ss[i] = ROTL64(state(i, idx), rot_amount[5 * i + idx]);
      // }

      // khi + iota
      state(lane5_inv, 0) = (RC[j] * (lane5 == 0)) ^ ss[0] ^ ((~ss[1]) & ss[2]);
#pragma unroll
      for (int i = 1; i < 5; i++) {
        state(lane5_inv, i) = ss[i] ^ ((~ss[(i + 1) % 5]) & ss[(i + 2) % 5]);
      }
    }
  }
} // namespace icicle::pqc::ml_kem
