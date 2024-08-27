#include <stdbool.h>

#ifndef _BN254_G2CURVE_H
#define _BN254_G2CURVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct g2_projective_t g2_projective_t;
typedef struct g2_affine_t g2_affine_t;
typedef struct VecOpsConfig VecOpsConfig;

bool bn254_g2_eq(g2_projective_t* point1, g2_projective_t* point2);
void bn254_g2_to_affine(g2_projective_t* point, g2_affine_t* point_out);
void bn254_g2_generate_projective_points(g2_projective_t* points, int size);
void bn254_g2_generate_affine_points(g2_affine_t* points, int size);
int bn254_g2_affine_convert_montgomery(const g2_affine_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, g2_affine_t* d_out);
int bn254_g2_projective_convert_montgomery(const g2_projective_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, g2_projective_t* d_out);

#ifdef __cplusplus
}
#endif

#endif
