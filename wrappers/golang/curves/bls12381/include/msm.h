#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BLS12_381_MSM_H
#define _BLS12_381_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_381MSMCuda(const scalar_t* scalars, const affine_t* points, int count, MSMConfig* config, projective_t* out);
cudaError_t bls12_381PrecomputeMSMBases(affine_t* points, int bases_size, int precompute_factor, int _c, bool are_bases_on_device, DeviceContext* ctx, affine_t* output_bases);

#ifdef __cplusplus
}
#endif

#endif
