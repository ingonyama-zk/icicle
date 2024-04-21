#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BLS12_381__g2MSM_H
#define _BLS12_381__g2MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct _g2_projective_t _g2_projective_t;
typedef struct _g2_affine_t _g2_affine_t;
typedef struct MSMConfig MSMConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t bls12_381_g2_msm_cuda(const scalar_t* scalars,const  _g2_affine_t* points, int count, MSMConfig* config, _g2_projective_t* out);
cudaError_t bls12_381_g2_precompute_msm_bases_cuda(_g2_affine_t* points, int count, int precompute_factor, int _c, bool bases_on_device, DeviceContext* ctx, _g2_affine_t* out);

#ifdef __cplusplus
}
#endif

#endif
