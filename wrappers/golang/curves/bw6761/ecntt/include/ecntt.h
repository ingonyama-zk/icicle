#include <cuda_runtime.h>

#ifndef _BW6_761_ECNTT_H
#define _BW6_761_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NTTConfig NTTConfig;
typedef struct projective_t projective_t;

cudaError_t bw6_761ECNTTCuda(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif