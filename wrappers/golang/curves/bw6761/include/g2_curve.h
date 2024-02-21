#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BW6_761_G2CURVE_H
#define _BW6_761_G2CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

bool bw6_761G2Eq(g2_projective_t* point1, g2_projective_t* point2);
void bw6_761G2ToAffine(g2_projective_t* point, g2_affine_t* point_out);
void bw6_761G2GenerateProjectivePoints(g2_projective_t* points, int size);
void bw6_761G2GenerateAffinePoints(g2_affine_t* points, int size);
cudaError_t bw6_761G2AffineConvertMontgomery(g2_affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bw6_761G2ProjectiveConvertMontgomery(g2_projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
