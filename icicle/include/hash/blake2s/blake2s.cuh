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
#include "merkle-tree/merkle.cuh"

#include "hash/hash.cuh"
using namespace hash;

#include "matrix/matrix.cuh"
using matrix::Matrix;

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

    cudaError_t hash_2d(
      const Matrix<BYTE>* inputs,
      BYTE* states,
      unsigned int number_of_inputs,
      unsigned int output_len,
      uint64_t number_of_rows,
      const device_context::DeviceContext& ctx) const override;

    cudaError_t compress_and_inject(
      const Matrix<BYTE>* matrices_to_inject,
      unsigned int number_of_inputs,
      uint64_t number_of_rows,
      const BYTE* prev_layer,
      BYTE* next_layer,
      unsigned int digest_elements,
      const device_context::DeviceContext& ctx) const override;

    Blake2s() : Hasher<BYTE, BYTE>(BLAKE2S_STATE_SIZE * 4, BLAKE2S_STATE_SIZE * 4, BLAKE2S_STATE_SIZE * 4, 0) {}
  };

  extern "C" {
  cudaError_t
  cuda_blake2s_hash_batch(BYTE* key, WORD keylen, BYTE* in, WORD inlen, BYTE* out, WORD output_len, WORD n_batch);

  cudaError_t blake2s_mmcs_commit_cuda(
    const Matrix<BYTE>* leaves,
    unsigned int number_of_inputs,
    BYTE* digests,
    const merkle_tree::TreeBuilderConfig& tree_config);
  }

} // namespace blake2s