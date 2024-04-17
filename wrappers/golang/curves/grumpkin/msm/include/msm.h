#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _GRUMPKIN_MSM_H
#define _GRUMPKIN_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct MSMConfig MSMConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t grumpkinMSMCuda(const scalar_t* scalars,const  affine_t* points, int count, MSMConfig* config, projective_t* out);
cudaError_t grumpkinPrecomputeMSMBases(affine_t* points, int count, int precompute_factor, int _c, bool bases_on_device, DeviceContext* ctx, affine_t* out);

#ifdef __cplusplus
}
#endif

#endif
