#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BABYBEAREXTENSION_NTT_H
#define _BABYBEAREXTENSION_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;


cudaError_t babybearExtensionNTTCuda(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);


#ifdef __cplusplus
}
#endif

#endif