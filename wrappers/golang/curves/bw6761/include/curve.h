#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BW6_761_CURVE_H
#define _BW6_761_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct DeviceContext DeviceContext;

bool bw6_761Eq(projective_t* point1, projective_t* point2);
void bw6_761ToAffine(projective_t* point, affine_t* point_out);
void bw6_761GenerateProjectivePoints(projective_t* points, int size);
void bw6_761GenerateAffinePoints(affine_t* points, int size);
cudaError_t bw6_761AffineConvertMontgomery(affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bw6_761ProjectiveConvertMontgomery(projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
