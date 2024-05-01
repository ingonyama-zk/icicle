#pragma once

#ifndef HASH_API_H
#define HASH_API_H

#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"
#include "hash/keccak/keccak.cuh"

extern "C" cudaError_t
  keccak256_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, keccak::KeccakConfig& config);

extern "C" cudaError_t
  keccak512_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, keccak::KeccakConfig& config);

#endif