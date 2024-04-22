#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BABYBEAR_EXTENSION_VEC_OPS_H
#define _BABYBEAR_EXTENSION_VEC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t babybear_extension_mul_cuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t babybear_extension_add_cuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t babybear_extension_sub_cuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t babybear_extension_transpose_matrix_cuda(
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
