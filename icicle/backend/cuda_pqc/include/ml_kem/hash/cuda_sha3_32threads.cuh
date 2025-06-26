#pragma once

#include "ml_kem/hash/cuda_hash_consts.cuh"
#include "ml_kem/hash/cuda_sha3_5threads.cuh"

namespace icicle::pqc::ml_kem {
  template <
    const int INPUT_LEN = 32,
    const int RATE,
    const uint32_t PAD_DELIM,
    const uint64_t PAD_SUFFIX,
    // USE_LDCS is a cache hint that reads from global memory in streaming mode,
    // meaning we don't expect to reuse the data many times. This is useful for
    // prefetching data that will only be read once.
    bool USE_LDCS = true>
  __device__ __forceinline__ uint64_t absorb(const uint8_t d[INPUT_LEN], uint32_t extra_input)
  {
    const int lane = threadIdx.x % 32;
    // load the random data, optionally using __ldcg to bypass L1 cache
    uint64_t s;
    if constexpr (USE_LDCS) {
      s = (lane < (INPUT_LEN / 8)) ? __ldcs((const uint64_t*)d + lane) : 0;
    } else {
      s = (lane < (INPUT_LEN / 8)) ? ((const uint64_t*)d)[lane] : 0;
    }
    // add the padding delimiter
    s |= (lane == (INPUT_LEN / 8)) ? (PAD_DELIM | extra_input) : 0;
    // add the padding suffix
    s |= (lane == (RATE / 8) - 1) ? PAD_SUFFIX : 0;
    return s;
  }

  // Version that takes two input arrays, each of size INPUT_LEN/2
  template <
    const int INPUT_LEN = 64,
    const int RATE,
    const uint32_t PAD_DELIM,
    const uint64_t PAD_SUFFIX,
    bool USE_LDCS = false>
  __device__ __forceinline__ uint64_t
  absorb_dual(const uint8_t d1[INPUT_LEN / 2], const uint8_t d2[INPUT_LEN / 2], uint32_t extra_input)
  {
    const int lane = threadIdx.x % 32;
    // load the random data, optionally using __ldcg to bypass L1 cache
    uint64_t s;
    if constexpr (USE_LDCS) {
      s = (lane < (INPUT_LEN / 16))  ? __ldcs((const uint64_t*)d1 + lane)
          : (lane < (INPUT_LEN / 8)) ? __ldcs((const uint64_t*)d2 + (lane - (INPUT_LEN / 16)))
                                     : 0;
    } else {
      s = (lane < (INPUT_LEN / 16))  ? ((const uint64_t*)d1)[lane]
          : (lane < (INPUT_LEN / 8)) ? ((const uint64_t*)d2)[lane - (INPUT_LEN / 16)]
                                     : 0;
    }
    // add the padding delimiter
    s |= (lane == (INPUT_LEN / 8)) ? (PAD_DELIM | extra_input) : 0;
    // add the padding suffix
    s |= (lane == (RATE / 8) - 1) ? PAD_SUFFIX : 0;
    return s;
  }

  template <const int RATE>
  __device__ __forceinline__ uint64_t absorb_intermediate(const uint8_t d[RATE])
  {
    const int lane = threadIdx.x % 32;
    return (lane < (RATE / 8)) ? ((const uint64_t*)d)[lane] : 0;
  }

  template <const int RATE>
  __device__ __forceinline__ uint64_t absorb_intermediate_dual(const uint8_t d1[32], const uint8_t d2[RATE])
  {
    const int lane = threadIdx.x % 32;
    return (lane < (32 / 8))     ? ((const uint64_t*)d1)[lane]
           : (lane < (RATE / 8)) ? ((const uint64_t*)d2)[lane - (32 / 8)]
                                 : 0;
  }

  __device__ uint64_t keccakf(uint64_t s)
  {
    const int lane = threadIdx.x % 32;
    uint64_t* cs = sha3_state_raw + MAX_HASHES_PER_WRAP * 25 * (threadIdx.x / 32);
    if (lane < 25) {
      for (int i = 0; i < 24; i++) {
        cs[lane] = s;

        // theta
        if (lane < 5) {
          const int col = lane % 5;
          uint64_t c = s;
          c ^= cs[col + 5];
          c ^= cs[col + 10];
          c ^= cs[col + 15];
          c ^= cs[col + 20];
          cs[col] = c;
        }

        const int next_col = (lane + 1) % 5;
        const int prev_col = (lane + 4) % 5;

        uint64_t c_minus_1 = cs[prev_col];
        uint64_t c_plus_1 = cs[next_col];
        uint64_t d = c_minus_1 ^ ROTL1(c_plus_1);
        s ^= d;

        // rho
        s = ROTL64(s, rot_amount[lane]);

        // phi
        const int col = lane % 5;
        const int row = lane / 5;
        const int dest = row + 5 * ((3 * row + 2 * col) % 5); // 6 -> 1
        cs[dest] = s;

        // khi + iota
        const int row_start = lane - (lane % 5);
        const uint64_t s1 = cs[((lane + 1) % 5) + row_start];
        const uint64_t s2 = cs[((lane + 2) % 5) + row_start];
        s = (cs[lane] ^ (RC[i] * (lane == 0))) ^ ((~s1) & s2);
      }
    }
    __syncwarp();
    return s;
  }
} // namespace icicle::pqc::ml_kem
