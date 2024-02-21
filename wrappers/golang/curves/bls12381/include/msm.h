#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BLS12_381_MSM_H
#define _BLS12_381_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_381MSMCuda(scalar_t* scalars, affine_t* points, int count, MSMConfig* config, projective_t* out);

#ifdef __cplusplus
}
#endif

#endif
