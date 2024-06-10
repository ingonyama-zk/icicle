#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BW6_761_MSM_H
#define _BW6_761_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct MSMConfig MSMConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bw6_761_msm_cuda(const scalar_t* scalars,const  affine_t* points, int count, MSMConfig* config, projective_t* out);
cudaError_t bw6_761_precompute_msm_bases_cuda(affine_t* points, int msm_size, MSMConfig* config, affine_t* out);

#ifdef __cplusplus
}
#endif

#endif
