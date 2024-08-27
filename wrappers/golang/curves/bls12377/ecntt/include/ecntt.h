#include <cuda_runtime.h>

#ifndef _BLS12_377_ECNTT_H
#define _BLS12_377_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NTTConfig NTTConfig;
typedef struct projective_t projective_t;

int bls12_377_ecntt(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif