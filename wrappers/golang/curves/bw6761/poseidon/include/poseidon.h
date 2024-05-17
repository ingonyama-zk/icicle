#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BW6_761_POSEIDON_H
#define _BW6_761_POSEIDON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct PoseidonConfig PoseidonConfig;
typedef struct DeviceContext DeviceContext;
typedef struct PoseidonConstants PoseidonConstants;


cudaError_t bw6_761_poseidon_hash_cuda(const scalar_t* input, scalar_t* output, int number_of_states, int arity, PoseidonConstants* constants, PoseidonConfig* config);
cudaError_t bw6_761_create_optimized_poseidon_constants_cuda(int arity, int full_rounds_halfs, int partial_rounds, const scalar_t* constants, DeviceContext* ctx, PoseidonConstants* poseidon_constants);
cudaError_t bw6_761_init_optimized_poseidon_constants_cuda(int arity, DeviceContext* ctx, PoseidonConstants* constants);

#ifdef __cplusplus
}
#endif

#endif