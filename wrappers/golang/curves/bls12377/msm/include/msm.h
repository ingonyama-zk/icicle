#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BLS12_377_MSM_H
#define _BLS12_377_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct MSMConfig MSMConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bls12_377_msm_cuda(const scalar_t* scalars,const  affine_t* points, int count, MSMConfig* config, projective_t* out);
cudaError_t bls12_377_precompute_msm_bases_cuda(affine_t* points, int count, int precompute_factor, int _c, bool bases_on_device, DeviceContext* ctx, affine_t* out);

#ifdef __cplusplus
}
#endif

#endif
