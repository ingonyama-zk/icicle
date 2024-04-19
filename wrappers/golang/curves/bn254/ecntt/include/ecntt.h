#include <cuda_runtime.h>
#include "../../../include/types.h"

#ifndef _BN254_ECNTT_H
#define _BN254_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bn254ECNTTCuda(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif