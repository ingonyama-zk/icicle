#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BLS12_381_NTT_H
#define _BLS12_381_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bls12_381NTTCuda(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
cudaError_t bls12_381InitializeDomain(scalar_t* primitive_root, DeviceContext* ctx, bool fast_twiddles);
cudaError_t bls12_381ReleaseDomain(DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif