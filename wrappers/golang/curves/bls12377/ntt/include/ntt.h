#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BLS12_377_NTT_H
#define _BLS12_377_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bls12_377NTTCuda(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
cudaError_t bls12_377InitializeDomain(scalar_t* primitive_root, DeviceContext* ctx, bool fast_twiddles);
cudaError_t bls12_377ReleaseDomain(DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif