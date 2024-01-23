#include "curves/curve_config.cuh"
#include "projective.cuh"
#include <cuda.h>
#include "utils/utils.h"

#define projective_t  curve_config::projective_t // TODO: global to avoid lengthy texts
#define affine_t      curve_config::affine_t
#define point_field_t curve_config::point_field_t

extern "C" bool CONCAT_EXPAND(CURVE, Eq)(projective_t* point1, projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == point_field_t::zero()) && (point1->y == point_field_t::zero()) &&
           (point1->z == point_field_t::zero())) &&
         !((point2->x == point_field_t::zero()) && (point2->y == point_field_t::zero()) &&
           (point2->z == point_field_t::zero()));
}

extern "C" void CONCAT_EXPAND(CURVE, ToAffine)(projective_t* point, affine_t* point_out)
{
  *point_out = projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, GenerateProjectivePoints)(projective_t* points, int size)
{
  projective_t::RandHostMany(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, GenerateAffinePoints)(affine_t* points, int size)
{
  projective_t::RandHostManyAffine(points, size);
}

#if defined(G2_DEFINED)

#define g2_projective_t  curve_config::g2_projective_t
#define g2_affine_t      curve_config::g2_affine_t
#define g2_point_field_t curve_config::g2_point_field_t

extern "C" bool CONCAT_EXPAND(CURVE, G2Eq)(g2_projective_t* point1, g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == g2_point_field_t::zero()) && (point1->y == g2_point_field_t::zero()) &&
           (point1->z == g2_point_field_t::zero())) &&
         !((point2->x == g2_point_field_t::zero()) && (point2->y == g2_point_field_t::zero()) &&
           (point2->z == g2_point_field_t::zero()));
}

extern "C" void CONCAT_EXPAND(CURVE, G2ToAffine)(g2_projective_t* point, g2_affine_t* point_out)
{
  *point_out = g2_projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, G2GenerateProjectivePoints)(g2_projective_t* points, int size)
{
  g2_projective_t::RandHostMany(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, G2GenerateAffinePoints)(g2_affine_t* points, int size)
{
  g2_projective_t::RandHostManyAffine(points, size);
}

#endif
