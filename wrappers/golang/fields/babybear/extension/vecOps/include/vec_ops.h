#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BABYBEAREXTENSION_VEC_OPS_H
#define _BABYBEAREXTENSION_VEC_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;
typedef struct DeviceContext DeviceContext;

cudaError_t babybearExtension_mul_cuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t babybearExtension_add_cuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t babybearExtensionsub_cuda(
  scalar_t* vec_a,
  scalar_t* vec_b,
  int n,
  VecOpsConfig* config,
  scalar_t* result
);

cudaError_t babybearExtensiontranspose_matrix_cuda(
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
