#include <cuda_runtime.h>
#include "../../../include/types.h"
#include <stdbool.h>

#ifndef _BLS12_381_ECNTT_H
#define _BLS12_381_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_381ECNTTCuda(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif