#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BLS12_377_CURVE_H
#define _BLS12_377_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

// G1
bool bls12_377Eq(projective_t* point1, projective_t* point2);
void bls12_377ToAffine(projective_t* point, affine_t* point_out);
void bls12_377GenerateProjectivePoints(projective_t* points, int size);
void bls12_377GenerateAffinePoints(affine_t* points, int size);
cudaError_t bls12_377AffineConvertMontgomery(affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bls12_377ProjectiveConvertMontgomery(projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

// G2
bool bls12_377G2Eq(g2_projective_t* point1, g2_projective_t* point2);
void bls12_377G2ToAffine(g2_projective_t* point, g2_affine_t* point_out);
void bls12_377G2GenerateProjectivePoints(g2_projective_t* points, int size);
void bls12_377G2GenerateAffinePoints(g2_affine_t* points, int size);
cudaError_t bls12_377G2AffineConvertMontgomery(g2_affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bls12_377G2ProjectiveConvertMontgomery(g2_projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
