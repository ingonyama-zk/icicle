#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BW6_761_MSM_H
#define _BW6_761_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bw6_761MSMCuda(scalar_t* scalars, affine_t* points, int count, MSMConfig* config, projective_t* out);

#ifdef __cplusplus
}
#endif

#endif
