#include <cuda_runtime.h>

#ifndef _TYPES_H
#define _TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct g2_projective_t g2_projective_t;
typedef struct affine_t affine_t;
typedef struct g2_affine_t g2_affine_t;

typedef struct MSMConfig MSMConfig;
typedef struct NTTConfig NTTConfig;
typedef struct VecOpsConfig VecOpsConfig;
typedef struct DeviceContext DeviceContext;

typedef cudaError_t cudaError_t;

#ifdef __cplusplus
}
#endif

#endif