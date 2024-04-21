#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BN254_MSM_H
#define _BN254_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct MSMConfig MSMConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bn254MSMCuda(const scalar_t* scalars,const  affine_t* points, int count, MSMConfig* config, projective_t* out);
cudaError_t bn254PrecomputeMSMBases(affine_t* points, int count, int precompute_factor, int _c, bool bases_on_device, DeviceContext* ctx, affine_t* out);

#ifdef __cplusplus
}
#endif

#endif
