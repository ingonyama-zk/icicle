#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BN254_CURVE_H
#define _BN254_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

bool bn254Eq(projective_t* point1, projective_t* point2);
void bn254ToAffine(projective_t* point, affine_t* point_out);
void bn254GenerateProjectivePoints(projective_t* points, int size);
void bn254GenerateAffinePoints(affine_t* points, int size);
cudaError_t bn254AffineConvertMontgomery(affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bn254G2AffineConvertMontgomery(g2_affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bn254ProjectiveConvertMontgomery(projective_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bn254G2ProjectiveConvertMontgomery(g2_projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
