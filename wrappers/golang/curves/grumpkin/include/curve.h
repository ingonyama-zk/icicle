#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _GRUMPKIN_CURVE_H
#define _GRUMPKIN_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct DeviceContext DeviceContext;

bool grumpkinEq(projective_t* point1, projective_t* point2);
void grumpkinToAffine(projective_t* point, affine_t* point_out);
void grumpkinGenerateProjectivePoints(projective_t* points, int size);
void grumpkinGenerateAffinePoints(affine_t* points, int size);
cudaError_t grumpkinAffineConvertMontgomery(affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t grumpkinProjectiveConvertMontgomery(projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
