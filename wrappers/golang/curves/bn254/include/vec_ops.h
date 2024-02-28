#include <cuda_runtime.h>
#include "../../include/types.h"

#ifndef _BN254_VEC_OPS_H
#define _BN254_VEC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bn254MulCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bn254AddCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bn254SubCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

#ifdef __cplusplus
}
#endif

#endif
