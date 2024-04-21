#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BLS12_377__g2CURVE_H
#define _BLS12_377__g2CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _g2_projective_t _g2_projective_t;
typedef struct _g2_affine_t _g2_affine_t;
typedef struct DeviceContext DeviceContext;

bool bls12_377_g2_eq(_g2_projective_t* point1, _g2_projective_t* point2);
void bls12_377_g2_to_affine(_g2_projective_t* point, _g2_affine_t* point_out);
void bls12_377_g2_generate_projective_points(_g2_projective_t* points, int size);
void bls12_377_g2_generate_affine_points(_g2_affine_t* points, int size);
cudaError_t bls12_377_g2_affine_convert_montgomery(_g2_affine_t* points, size_t n, bool is_into, DeviceContext* ctx);
cudaError_t bls12_377_g2_projective_convert_montgomery(_g2_projective_t* points, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
