#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BLS12_377_NTT_H
#define _BLS12_377_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_377NTTCuda(scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
cudaError_t bls12_377InitializeDomain(scalar_t* primitive_root, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
