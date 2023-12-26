#include <cuda_runtime.h>

// #define G2_DEFINED
// #include "../../../../../icicle/curves/curve_config.cuh"

#ifndef _TYPES_H
#define _TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

// typedef curve_config::scalar_t scalar_t;
// typedef curve_config::projective_t projective_t;
// typedef curve_config::g2_projective_t g2_projective_t;
// typedef curve_config::affine_t affine_t;
// typedef curve_config::g2_affine_t g2_affine_t;

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct g2_projective_t g2_projective_t;
typedef struct affine_t affine_t;
typedef struct g2_affine_t g2_affine_t;

typedef struct MSMConfig MSMConfig;
typedef struct NTTConfig NTTConfig;
typedef struct DeviceContext DeviceContext;

typedef cudaError_t cudaError_t;

#ifdef __cplusplus
}
#endif

#endif