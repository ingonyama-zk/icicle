#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BLS12_377_G2MSM_H
#define _BLS12_377_G2MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_377G2MSMCuda(scalar_t* scalars, g2_affine_t* points, int count, MSMConfig* config, g2_projective_t* out);

#ifdef __cplusplus
}
#endif

#endif
