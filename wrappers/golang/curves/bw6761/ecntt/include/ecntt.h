#include <cuda_runtime.h>
#include "../../../include/types.h"

#ifndef _BW6_761_ECNTT_H
#define _BW6_761_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bw6_761ECNTTCuda(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif