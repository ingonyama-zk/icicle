#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BN254_MSM_H
#define _BN254_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bn254MSMCuda(scalar_t* scalars, affine_t* points, int count, MSMConfig* config, projective_t* out);

#ifdef __cplusplus
}
#endif

#endif
