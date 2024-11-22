#pragma once

#ifndef HASH_API_H
#define HASH_API_H

#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"
#include "hash/keccak/keccak.cuh"
#include "hash/blake2s/blake2s.cuh"
#include "merkle-tree/merkle.cuh"

/* KECCAK */

extern "C" cudaError_t
  keccak256_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, keccak::HashConfig& config);

extern "C" cudaError_t
  keccak512_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, keccak::HashConfig& config);

extern "C" cudaError_t build_keccak256_merkle_tree_cuda(
  const uint8_t* leaves,
  uint64_t* digests,
  unsigned int height,
  unsigned int input_block_len,
  const merkle_tree::TreeBuilderConfig& tree_config);

extern "C" cudaError_t build_keccak512_merkle_tree_cuda(
  const uint8_t* leaves,
  uint64_t* digests,
  unsigned int height,
  unsigned int input_block_len,
  const merkle_tree::TreeBuilderConfig& tree_config);

/* BLAKE2S */

extern "C" cudaError_t blake2s_cuda(
  BYTE* input, BYTE* output, WORD number_of_blocks, WORD input_block_size, WORD output_block_size, blake2s::HashConfig& config)

extern "C" cudaError_t build_blake2s_merkle_tree_cuda(
  const uint8_t* leaves,
  uint64_t* digests,
  unsigned int height,
  unsigned int input_block_len,
  const merkle_tree::TreeBuilderConfig& tree_config);
#endif