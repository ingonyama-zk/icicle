#include <cuda_runtime.h>
#include "../../../include/types.h"

#ifndef _BLS12_377_ECNTT_H
#define _BLS12_377_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_377ECNTTCuda(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif