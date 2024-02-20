#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BLS12_381_CURVE_H
#define _BLS12_381_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

// G1
bool bls12_381Eq(projective_t* point1, projective_t* point2);
void bls12_381ToAffine(projective_t* point, affine_t* point_out);
void bls12_381GenerateProjectivePoints(projective_t* points, int size);
void bls12_381GenerateAffinePoints(affine_t* points, int size);
cudaError_t bls12_381AffineConvertMontgomery(affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bls12_381ProjectiveConvertMontgomery(projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

// G2
bool bls12_381G2Eq(g2_projective_t* point1, g2_projective_t* point2);
void bls12_381G2ToAffine(g2_projective_t* point, g2_affine_t* point_out);
void bls12_381G2GenerateProjectivePoints(g2_projective_t* points, int size);
void bls12_381G2GenerateAffinePoints(g2_affine_t* points, int size);
cudaError_t bls12_381G2AffineConvertMontgomery(g2_affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bls12_381G2ProjectiveConvertMontgomery(g2_projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
