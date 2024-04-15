#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BN254_MSM_H
#define _BN254_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bn254MSMCuda(const scalar_t* scalars, const affine_t* points, int count, MSMConfig* config, projective_t* out);
cudaError_t bn254PrecomputeMSMBases(affine_t* points, int bases_size, int precompute_factor, int _c, bool are_bases_on_device, DeviceContext* ctx, affine_t* output_bases);

#ifdef __cplusplus
}
#endif

#endif
