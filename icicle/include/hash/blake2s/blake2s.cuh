/*
 * blake2b.cuh CUDA Implementation of BLAKE2B Hashing
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is released into the Public Domain.
 */

#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"
using namespace hash;

namespace blake2s {

  typedef unsigned char BYTE;
  typedef unsigned int WORD;
  typedef unsigned long long LONG;

#define BLAKE2S_ROUNDS       10
#define BLAKE2S_BLOCK_LENGTH 64
#define BLAKE2S_CHAIN_SIZE   8
#define BLAKE2S_CHAIN_LENGTH (BLAKE2S_CHAIN_SIZE * sizeof(uint32_t))
#define BLAKE2S_STATE_SIZE   16
#define BLAKE2S_STATE_LENGTH (BLAKE2S_STATE_SIZE * sizeof(uint32_t))

  class Blake2s : public Hasher<BYTE, BYTE>
  {
  public:
    cudaError_t run_hash_many_kernel(
      const BYTE* input,
      BYTE* output,
      WORD number_of_states,
      WORD input_len,
      WORD output_len,
      const device_context::DeviceContext& ctx) const override;

    Blake2s() : Hasher<BYTE, BYTE>(BLAKE2S_STATE_SIZE, BLAKE2S_STATE_SIZE, BLAKE2S_STATE_SIZE, 0) {}
  };

  extern "C" {
  cudaError_t
  cuda_blake2s_hash_batch(BYTE* key, WORD keylen, BYTE* in, WORD inlen, BYTE* out, WORD output_len, WORD n_batch);
  }
} // namespace blake2s