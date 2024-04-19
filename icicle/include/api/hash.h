#pragma once

#ifndef HASH_API_H
#define HASH_API_H

#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"
#include "hash/keccak/keccak.cuh"

extern "C" cudaError_t
  Keccak256(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig config);

extern "C" cudaError_t
  Keccak512(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig config);

#endif