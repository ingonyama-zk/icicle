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

extern "C" void CONCAT_EXPAND(CURVE, generate_projective_points)(projective_t* points, int size)
{
  projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, generate_affine_points)(affine_t* points, int size)
{
  projective_t::rand_host_many_affine(points, size);
}

/********************************** G2 **********************************/
extern "C" bool CONCAT_EXPAND(CURVE, g2_eq)(g2_projective_t* point1, g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == g2_point_field_t::zero()) && (point1->y == g2_point_field_t::zero()) &&
           (point1->z == g2_point_field_t::zero())) &&
         !((point2->x == g2_point_field_t::zero()) && (point2->y == g2_point_field_t::zero()) &&
           (point2->z == g2_point_field_t::zero()));
}

extern "C" void CONCAT_EXPAND(CURVE, g2_to_affine)(g2_projective_t* point, g2_affine_t* point_out)
{
  *point_out = g2_projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_projective_points)(g2_projective_t* points, int size)
{
  g2_projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_affine_points)(g2_affine_t* points, int size)
{
  g2_projective_t::rand_host_many_affine(points, size);
}
