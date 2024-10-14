#include <stdbool.h>

#ifndef _GRUMPKIN_CURVE_H
  #define _GRUMPKIN_CURVE_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct VecOpsConfig VecOpsConfig;

bool grumpkin_eq(projective_t* point1, projective_t* point2);
void grumpkin_to_affine(projective_t* point, affine_t* point_out);
void grumpkin_from_affine(affine_t* point, projective_t* point_out);
void grumpkin_generate_projective_points(projective_t* points, int size);
void grumpkin_generate_affine_points(affine_t* points, int size);
int grumpkin_affine_convert_montgomery(
  const affine_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, affine_t* d_out);
int grumpkin_projective_convert_montgomery(
  const projective_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, projective_t* d_out);

  #ifdef __cplusplus
}
  #endif

#endif
