#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BW6_761_G2MSM_H
#define _BW6_761_G2MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bw6_761G2MSMCuda(scalar_t* scalars, g2_affine_t* points, int count, MSMConfig* config, g2_projective_t* out);

#ifdef __cplusplus
}
#endif

#endif
