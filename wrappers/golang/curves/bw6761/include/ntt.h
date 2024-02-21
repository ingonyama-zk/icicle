#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BW6_761_NTT_H
#define _BW6_761_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bw6_761NTTCuda(scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
cudaError_t bw6_761InitializeDomain(scalar_t* primitive_root, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
