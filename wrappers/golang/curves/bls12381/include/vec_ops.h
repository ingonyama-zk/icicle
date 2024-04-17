#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BLS12_381_VEC_OPS_H
#define _BLS12_381_VEC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t bls12_381MulCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bls12_381AddCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bls12_381SubCuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t bls12_381TransposeMatrix(
  scalar_t* mat_in,
  int row_size,
  int column_size,
  scalar_t* mat_out,
  DeviceContext* ctx,
  bool on_device,
  bool is_async
);

#ifdef __cplusplus
}
#endif

#endif
