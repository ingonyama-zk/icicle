#include <stdbool.h>

#ifndef _BN254_CURVE_H
#define _BN254_CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct VecOpsConfig VecOpsConfig;

bool bn254_eq(projective_t* point1, projective_t* point2);
void bn254_to_affine(projective_t* point, affine_t* point_out);
void bn254_generate_projective_points(projective_t* points, int size);
void bn254_generate_affine_points(affine_t* points, int size);
int bn254_affine_convert_montgomery(const affine_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, affine_t* d_out);
int bn254_projective_convert_montgomery(const projective_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, projective_t* d_out);

#ifdef __cplusplus
}
#endif

#endif
