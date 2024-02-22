#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BN254_G2CURVE_H
#define _BN254_G2CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

bool bn254G2Eq(g2_projective_t* point1, g2_projective_t* point2);
void bn254G2ToAffine(g2_projective_t* point, g2_affine_t* point_out);
void bn254G2GenerateProjectivePoints(g2_projective_t* points, int size);
void bn254G2GenerateAffinePoints(g2_affine_t* points, int size);
cudaError_t bn254G2AffineConvertMontgomery(g2_affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bn254G2ProjectiveConvertMontgomery(g2_projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
