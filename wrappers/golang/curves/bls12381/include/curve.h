#include <stdbool.h>

#ifndef _BLS12_381_CURVE_H
#define _BLS12_381_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct VecOpsConfig VecOpsConfig;

bool bls12_381_eq(projective_t* point1, projective_t* point2);
void bls12_381_ecadd(projective_t* point, projective_t* point2, projective_t* res); 
void bls12_381_ecsub(projective_t* point, projective_t* point2, projective_t* res); 
void bls12_381_to_affine(projective_t* point, affine_t* point_out);
void bls12_381_from_affine(affine_t* point, projective_t* point_out);
void bls12_381_generate_projective_points(projective_t* points, int size);
void bls12_381_generate_affine_points(affine_t* points, int size);
int bls12_381_affine_convert_montgomery(const affine_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, affine_t* d_out);
int bls12_381_projective_convert_montgomery(const projective_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, projective_t* d_out);

#ifdef __cplusplus
}
#endif

#endif
