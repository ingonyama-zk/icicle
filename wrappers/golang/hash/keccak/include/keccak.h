#include <stdint.h>
#include <cuda_runtime.h>

#ifndef _KECCAK_HASH_H
#define _KECCAK_HASH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HashConfig HashConfig;

cudaError_t keccak256_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, HashConfig* config);
cudaError_t keccak512_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, HashConfig* config);

#ifdef __cplusplus
}
#endif

#endif
