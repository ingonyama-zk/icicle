#pragma once
#ifndef KECCAK_KERNELS_H
#define KECCAK_KERNELS_H

#include <cstdint>
#include "gpu-utils/modifiers.cuh"

namespace keccak {
  using u64 = uint64_t;

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

#define TH_ELT(t, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4)                                                              \
  {                                                                                                                    \
    t = ROTL64((d0 ^ d1 ^ d2 ^ d3 ^ d4), 1) ^ (c0 ^ c1 ^ c2 ^ c3 ^ c4);                                                \
  }

#define THETA(                                                                                                         \
  s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, s22, s23, s24, s30, s31, s32, s33, s34, s40, s41, s42,   \
  s43, s44)                                                                                                            \
  {                                                                                                                    \
    TH_ELT(t0, s40, s41, s42, s43, s44, s10, s11, s12, s13, s14);                                                      \
    TH_ELT(t1, s00, s01, s02, s03, s04, s20, s21, s22, s23, s24);                                                      \
    TH_ELT(t2, s10, s11, s12, s13, s14, s30, s31, s32, s33, s34);                                                      \
    TH_ELT(t3, s20, s21, s22, s23, s24, s40, s41, s42, s43, s44);                                                      \
    TH_ELT(t4, s30, s31, s32, s33, s34, s00, s01, s02, s03, s04);                                                      \
    s00 ^= t0;                                                                                                         \
    s01 ^= t0;                                                                                                         \
    s02 ^= t0;                                                                                                         \
    s03 ^= t0;                                                                                                         \
    s04 ^= t0;                                                                                                         \
                                                                                                                       \
    s10 ^= t1;                                                                                                         \
    s11 ^= t1;                                                                                                         \
    s12 ^= t1;                                                                                                         \
    s13 ^= t1;                                                                                                         \
    s14 ^= t1;                                                                                                         \
                                                                                                                       \
    s20 ^= t2;                                                                                                         \
    s21 ^= t2;                                                                                                         \
    s22 ^= t2;                                                                                                         \
    s23 ^= t2;                                                                                                         \
    s24 ^= t2;                                                                                                         \
                                                                                                                       \
    s30 ^= t3;                                                                                                         \
    s31 ^= t3;                                                                                                         \
    s32 ^= t3;                                                                                                         \
    s33 ^= t3;                                                                                                         \
    s34 ^= t3;                                                                                                         \
                                                                                                                       \
    s40 ^= t4;                                                                                                         \
    s41 ^= t4;                                                                                                         \
    s42 ^= t4;                                                                                                         \
    s43 ^= t4;                                                                                                         \
    s44 ^= t4;                                                                                                         \
  }

#define RHOPI(                                                                                                         \
  s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, s22, s23, s24, s30, s31, s32, s33, s34, s40, s41, s42,   \
  s43, s44)                                                                                                            \
  {                                                                                                                    \
    t0 = ROTL64(s10, (uint64_t)1);                                                                                     \
    s10 = ROTL64(s11, (uint64_t)44);                                                                                   \
    s11 = ROTL64(s41, (uint64_t)20);                                                                                   \
    s41 = ROTL64(s24, (uint64_t)61);                                                                                   \
    s24 = ROTL64(s42, (uint64_t)39);                                                                                   \
    s42 = ROTL64(s04, (uint64_t)18);                                                                                   \
    s04 = ROTL64(s20, (uint64_t)62);                                                                                   \
    s20 = ROTL64(s22, (uint64_t)43);                                                                                   \
    s22 = ROTL64(s32, (uint64_t)25);                                                                                   \
    s32 = ROTL64(s43, (uint64_t)8);                                                                                    \
    s43 = ROTL64(s34, (uint64_t)56);                                                                                   \
    s34 = ROTL64(s03, (uint64_t)41);                                                                                   \
    s03 = ROTL64(s40, (uint64_t)27);                                                                                   \
    s40 = ROTL64(s44, (uint64_t)14);                                                                                   \
    s44 = ROTL64(s14, (uint64_t)2);                                                                                    \
    s14 = ROTL64(s31, (uint64_t)55);                                                                                   \
    s31 = ROTL64(s13, (uint64_t)45);                                                                                   \
    s13 = ROTL64(s01, (uint64_t)36);                                                                                   \
    s01 = ROTL64(s30, (uint64_t)28);                                                                                   \
    s30 = ROTL64(s33, (uint64_t)21);                                                                                   \
    s33 = ROTL64(s23, (uint64_t)15);                                                                                   \
    s23 = ROTL64(s12, (uint64_t)10);                                                                                   \
    s12 = ROTL64(s21, (uint64_t)6);                                                                                    \
    s21 = ROTL64(s02, (uint64_t)3);                                                                                    \
    s02 = t0;                                                                                                          \
  }

#define KHI(                                                                                                           \
  s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, s22, s23, s24, s30, s31, s32, s33, s34, s40, s41, s42,   \
  s43, s44)                                                                                                            \
  {                                                                                                                    \
    t0 = s00 ^ (~s10 & s20);                                                                                           \
    t1 = s10 ^ (~s20 & s30);                                                                                           \
    t2 = s20 ^ (~s30 & s40);                                                                                           \
    t3 = s30 ^ (~s40 & s00);                                                                                           \
    t4 = s40 ^ (~s00 & s10);                                                                                           \
    s00 = t0;                                                                                                          \
    s10 = t1;                                                                                                          \
    s20 = t2;                                                                                                          \
    s30 = t3;                                                                                                          \
    s40 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s01 ^ (~s11 & s21);                                                                                           \
    t1 = s11 ^ (~s21 & s31);                                                                                           \
    t2 = s21 ^ (~s31 & s41);                                                                                           \
    t3 = s31 ^ (~s41 & s01);                                                                                           \
    t4 = s41 ^ (~s01 & s11);                                                                                           \
    s01 = t0;                                                                                                          \
    s11 = t1;                                                                                                          \
    s21 = t2;                                                                                                          \
    s31 = t3;                                                                                                          \
    s41 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s02 ^ (~s12 & s22);                                                                                           \
    t1 = s12 ^ (~s22 & s32);                                                                                           \
    t2 = s22 ^ (~s32 & s42);                                                                                           \
    t3 = s32 ^ (~s42 & s02);                                                                                           \
    t4 = s42 ^ (~s02 & s12);                                                                                           \
    s02 = t0;                                                                                                          \
    s12 = t1;                                                                                                          \
    s22 = t2;                                                                                                          \
    s32 = t3;                                                                                                          \
    s42 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s03 ^ (~s13 & s23);                                                                                           \
    t1 = s13 ^ (~s23 & s33);                                                                                           \
    t2 = s23 ^ (~s33 & s43);                                                                                           \
    t3 = s33 ^ (~s43 & s03);                                                                                           \
    t4 = s43 ^ (~s03 & s13);                                                                                           \
    s03 = t0;                                                                                                          \
    s13 = t1;                                                                                                          \
    s23 = t2;                                                                                                          \
    s33 = t3;                                                                                                          \
    s43 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s04 ^ (~s14 & s24);                                                                                           \
    t1 = s14 ^ (~s24 & s34);                                                                                           \
    t2 = s24 ^ (~s34 & s44);                                                                                           \
    t3 = s34 ^ (~s44 & s04);                                                                                           \
    t4 = s44 ^ (~s04 & s14);                                                                                           \
    s04 = t0;                                                                                                          \
    s14 = t1;                                                                                                          \
    s24 = t2;                                                                                                          \
    s34 = t3;                                                                                                          \
    s44 = t4;                                                                                                          \
  }

#define IOTA(element, rc)                                                                                              \
  {                                                                                                                    \
    element ^= rc;                                                                                                     \
  }

  __device__ const u64 RC[24] = {0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
                                 0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
                                 0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
                                 0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
                                 0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
                                 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};

  __device__ void keccakf(u64 s[25])
  {
    u64 t0, t1, t2, t3, t4;

    for (int i = 0; i < 24; i++) {
      THETA(
        s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
        s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);
      RHOPI(
        s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
        s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);
      KHI(
        s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
        s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);
      IOTA(s[0], RC[i]);
    }
  }

  template <int C, int D>
  __global__ void keccak_hash_blocks(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output)
  {
    int bid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (bid >= number_of_blocks) { return; }

    const int r_bits = 1600 - C;
    const int r_bytes = r_bits / 8;
    const int d_bytes = D / 8;

    uint8_t* b_input = input + bid * input_block_size;
    uint8_t* b_output = output + bid * d_bytes;
    uint64_t state[25] = {}; // Initialize with zeroes

    int input_len = input_block_size;

    // absorb
    while (input_len >= r_bytes) {
      // #pragma unroll
      for (int i = 0; i < r_bytes; i += 8) {
        state[i / 8] ^= *(uint64_t*)(b_input + i);
      }
      keccakf(state);
      b_input += r_bytes;
      input_len -= r_bytes;
    }

    // last block (if any)
    uint8_t last_block[r_bytes];
    for (int i = 0; i < input_len; i++) {
      last_block[i] = b_input[i];
    }

    // pad 10*1
    last_block[input_len] = 1;
    for (int i = 0; i < r_bytes - input_len - 1; i++) {
      last_block[input_len + i + 1] = 0;
    }
    // last bit
    last_block[r_bytes - 1] |= 0x80;

    // #pragma unroll
    for (int i = 0; i < r_bytes; i += 8) {
      state[i / 8] ^= *(uint64_t*)(last_block + i);
    }
    keccakf(state);

#pragma unroll
    for (int i = 0; i < d_bytes; i += 8) {
      *(uint64_t*)(b_output + i) = state[i / 8];
    }
  }
} // namespace keccak

#endif