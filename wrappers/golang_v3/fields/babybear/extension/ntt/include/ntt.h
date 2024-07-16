#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BABYBEAR_EXTENSION_NTT_H
#define _BABYBEAR_EXTENSION_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;


int babybear_extension_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);


#ifdef __cplusplus
}
#endif

#endif