#include "icicle/utils/utils.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

// extern functions for FFI

/********************************** G1 **********************************/
extern "C" bool CONCAT_EXPAND(CURVE, eq)(projective_t* point1, projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == point_field_t::zero()) && (point1->y == point_field_t::zero()) &&
           (point1->z == point_field_t::zero())) &&
         !((point2->x == point_field_t::zero()) && (point2->y == point_field_t::zero()) &&
           (point2->z == point_field_t::zero()));
}

extern "C" void CONCAT_EXPAND(CURVE, to_affine)(projective_t* point, affine_t* point_out)
{
  *point_out = projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, from_affine)(affine_t* point, projective_t* point_out)
{
  *point_out = projective_t::from_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, generate_projective_points)(projective_t* points, int size)
{
  projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, generate_affine_points)(affine_t* points, int size)
{
  projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, ecsub)(projective_t* point1, projective_t* point2, projective_t* result)
{
  *result = *point1 - *point2;
}
extern "C" void CONCAT_EXPAND(CURVE, ecadd)(projective_t* point1, projective_t* point2, projective_t* result)
{
  *result = *point1 + *point2;
}
extern "C" void CONCAT_EXPAND(CURVE, mul_scalar)(projective_t* point, scalar_t* scalar, projective_t* result)
{
  *result = *point * *scalar;
}
/********************************** point_field_t **********************************/

extern "C" void CONCAT_EXPAND(CURVE, point_field_from_u32)(uint32_t val, point_field_t* res)
{
  *res = point_field_t::from(val);
}

extern "C" void CONCAT_EXPAND(CURVE, point_field_to_montgomery)(const point_field_t& scalar, point_field_t* res)
{
  *res = point_field_t::to_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(CURVE, point_field_from_montgomery)(const point_field_t& scalar, point_field_t* res)
{
  *res = point_field_t::from_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(FIELD, point_field_sub)(point_field_t* scalar1, point_field_t* scalar2, point_field_t* result)
{
  *result = *scalar1 - *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, point_field_add)(point_field_t* scalar1, point_field_t* scalar2, point_field_t* result)
{
  *result = *scalar1 + *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, point_field_mul)(point_field_t* scalar1, point_field_t* scalar2, point_field_t* result)
{
  *result = *scalar1 * *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, point_field_inv)(point_field_t* scalar1, point_field_t* result)
{
  *result = point_field_t::inverse(*scalar1);
}

/********************************** G2 **********************************/
#ifdef G2
extern "C" bool CONCAT_EXPAND(CURVE, g2_eq)(g2_projective_t* point1, g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == g2_point_field_t::zero()) && (point1->y == g2_point_field_t::zero()) &&
           (point1->z == g2_point_field_t::zero())) &&
         !((point2->x == g2_point_field_t::zero()) && (point2->y == g2_point_field_t::zero()) &&
           (point2->z == g2_point_field_t::zero()));
}

extern "C" void
CONCAT_EXPAND(CURVE, g2_ecsub)(g2_projective_t* point1, g2_projective_t* point2, g2_projective_t* result)
{
  *result = *point1 - *point2;
}

extern "C" void
CONCAT_EXPAND(CURVE, g2_ecadd)(g2_projective_t* point1, g2_projective_t* point2, g2_projective_t* result)
{
  *result = *point1 + *point2;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_mul_scalar)(g2_projective_t* point, scalar_t* scalar, g2_projective_t* result)
{
  *result = *point * *scalar;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_to_affine)(g2_projective_t* point, g2_affine_t* point_out)
{
  *point_out = g2_projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_from_affine)(g2_affine_t* point, g2_projective_t* point_out)
{
  *point_out = g2_projective_t::from_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_projective_points)(g2_projective_t* points, int size)
{
  g2_projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_affine_points)(g2_affine_t* points, int size)
{
  g2_projective_t::rand_host_many(points, size);
}

/********************************** g2_point_field_t **********************************/

extern "C" void CONCAT_EXPAND(CURVE, g2_point_field_from_u32)(uint32_t val, g2_point_field_t* res)
{
  *res = g2_point_field_t::from(val);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_point_field_to_montgomery)(const g2_point_field_t& scalar, g2_point_field_t* res)
{
  *res = g2_point_field_t::to_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_point_field_from_montgomery)(const g2_point_field_t& scalar, g2_point_field_t* res)
{
  *res = g2_point_field_t::from_montgomery(scalar);
}
extern "C" void CONCAT_EXPAND(FIELD, g2_point_field_sub)(g2_point_field_t* scalar1, g2_point_field_t* scalar2, g2_point_field_t* result)
{
  *result = *scalar1 - *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, g2_point_field_add)(g2_point_field_t* scalar1, g2_point_field_t* scalar2, g2_point_field_t* result)
{
  *result = *scalar1 + *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, g2_point_field_mul)(g2_point_field_t* scalar1, g2_point_field_t* scalar2, g2_point_field_t* result)
{
  *result = *scalar1 * *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, g2_point_field_inv)(g2_point_field_t* scalar1, g2_point_field_t* result)
{
  *result = g2_point_field_t::inverse(*scalar1);
}
#endif // G2
