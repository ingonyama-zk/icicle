#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BW6_761_G2MSM_H
#define _BW6_761_G2MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct g2_projective_t g2_projective_t;
typedef struct g2_affine_t g2_affine_t;
typedef struct MSMConfig MSMConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bw6_761_g2_msm_cuda(const scalar_t* scalars,const  g2_affine_t* points, int count, MSMConfig* config, g2_projective_t* out);
cudaError_t bw6_761_g2_precompute_msm_bases_cuda(g2_affine_t* points, int count, int precompute_factor, int _c, bool bases_on_device, DeviceContext* ctx, g2_affine_t* out);
cudaError_t bw6_761_g2_precompute_msm_points_cuda(g2_affine_t* points, int msm_size, MSMConfig* config, g2_affine_t* out);

#ifdef __cplusplus
}
#endif

#endif
