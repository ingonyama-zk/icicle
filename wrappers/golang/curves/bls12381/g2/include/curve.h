#include <stdbool.h>

#ifndef _BLS12_381_G2CURVE_H
  #define _BLS12_381_G2CURVE_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct g2_projective_t g2_projective_t;
typedef struct g2_affine_t g2_affine_t;
typedef struct VecOpsConfig VecOpsConfig;

bool bls12_381_g2_eq(g2_projective_t* point1, g2_projective_t* point2);
void bls12_381_g2_to_affine(g2_projective_t* point, g2_affine_t* point_out);
void bls12_381_g2_from_affine(g2_affine_t* point, g2_projective_t* point_out);
void bls12_381_g2_generate_projective_points(g2_projective_t* points, int size);
void bls12_381_g2_generate_affine_points(g2_affine_t* points, int size);
int bls12_381_g2_affine_convert_montgomery(
  const g2_affine_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, g2_affine_t* d_out);
int bls12_381_g2_projective_convert_montgomery(
  const g2_projective_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, g2_projective_t* d_out);

  #ifdef __cplusplus
}
  #endif

#endif
