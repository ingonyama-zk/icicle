#include <stdbool.h>

#ifndef _BN254_VEC_OPS_H
  #define _BN254_VEC_OPS_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;
typedef struct DeviceContext DeviceContext;

int bn254_vector_mul(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig* config, scalar_t* result);

int bn254_vector_add(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig* config, scalar_t* result);

int bn254_vector_sub(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig* config, scalar_t* result);

int bn254_matrix_transpose(scalar_t* mat_in, int row_size, int column_size, VecOpsConfig* config, scalar_t* mat_out);

  #ifdef __cplusplus
}
  #endif

#endif
