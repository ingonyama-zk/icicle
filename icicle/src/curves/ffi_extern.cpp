#include "icicle/utils/utils.h"
#include "icicle/curves/curve_config.h"
#include "icicle/fields/externs.h"

using namespace curve_config;

// extern functions for FFI

/********************************** G1 **********************************/
extern "C" bool CONCAT_EXPAND(ICICLE_FFI_PREFIX, projective_eq)(projective_t* point1, projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == point_field_t::zero()) && (point1->y == point_field_t::zero()) &&
           (point1->z == point_field_t::zero())) &&
         !((point2->x == point_field_t::zero()) && (point2->y == point_field_t::zero()) &&
           (point2->z == point_field_t::zero()));
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, ecsub)(projective_t* point1, projective_t* point2, projective_t* result)
{
  *result = *point1 - *point2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, ecadd)(projective_t* point1, projective_t* point2, projective_t* result)
{
  *result = *point1 + *point2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, mul_scalar)(projective_t* point, scalar_t* scalar, projective_t* result)
{
  *result = *point * *scalar;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, to_affine)(projective_t* point, affine_t* point_out)
{
  *point_out = point->to_affine();
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, from_affine)(affine_t* point, projective_t* point_out)
{
  *point_out = projective_t::from_affine(*point);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, generate_projective_points)(projective_t* points, int size)
{
  projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, generate_affine_points)(affine_t* points, int size)
{
  projective_t::rand_host_many(points, size);
}

ICICLE_DEFINE_FIELD_FFI_FUNCS(_base_field, point_field_t);

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, generator)(projective_t* result)
{
  *result = projective_t::generator();
}

extern "C" bool CONCAT_EXPAND(ICICLE_FFI_PREFIX, is_on_curve)(projective_t* point) { return point->is_on_curve(); }

/********************************** G2 **********************************/
#ifdef G2_ENABLED
extern "C" bool CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_projective_eq)(g2_projective_t* point1, g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == g2_point_field_t::zero()) && (point1->y == g2_point_field_t::zero()) &&
           (point1->z == g2_point_field_t::zero())) &&
         !((point2->x == g2_point_field_t::zero()) && (point2->y == g2_point_field_t::zero()) &&
           (point2->z == g2_point_field_t::zero()));
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_ecsub)(g2_projective_t* point1, g2_projective_t* point2, g2_projective_t* result)
{
  *result = *point1 - *point2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_ecadd)(g2_projective_t* point1, g2_projective_t* point2, g2_projective_t* result)
{
  *result = *point1 + *point2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_mul_scalar)(g2_projective_t* point, scalar_t* scalar, g2_projective_t* result)
{
  *result = *point * *scalar;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_to_affine)(g2_projective_t* point, g2_affine_t* point_out)
{
  *point_out = point->to_affine();
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_from_affine)(g2_affine_t* point, g2_projective_t* point_out)
{
  *point_out = g2_projective_t::from_affine(*point);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_generate_projective_points)(g2_projective_t* points, int size)
{
  g2_projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_generate_affine_points)(g2_affine_t* points, int size)
{
  g2_projective_t::rand_host_many(points, size);
}

ICICLE_DEFINE_FIELD_FFI_FUNCS(_g2_base_field, g2_point_field_t);

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_generator)(g2_projective_t* result)
{
  *result = g2_projective_t::generator();
}

extern "C" bool CONCAT_EXPAND(ICICLE_FFI_PREFIX, g2_is_on_curve)(g2_projective_t* point)
{
  return point->is_on_curve();
}
#endif // G2
