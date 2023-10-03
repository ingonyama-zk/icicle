#include "projective.cuh"
#include "../curves/curve_config.cuh"
#include <cuda.h>

#define projective_t curve_config::projective_t //TODO: global to avoid lengthy texts
#define point_field_t curve_config::point_field_t

extern "C" bool eq(projective_t* point1, projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == point_field_t::zero()) && (point1->y == point_field_t::zero()) &&
           (point1->z == point_field_t::zero())) &&
         !((point2->x == point_field_t::zero()) && (point2->y == point_field_t::zero()) &&
           (point2->z == point_field_t::zero()));
}

#if defined(G2_DEFINED)

#define projective_t curve_config::g2_projective_t
#define point_field_t curve_config::g2_point_field_t

extern "C" bool eq_g2(g2_projective_t* point1, g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == g2_point_field_t::zero()) && (point1->y == g2_point_field_t::zero()) &&
           (point1->z == g2_point_field_t::zero())) &&
         !((point2->x == g2_point_field_t::zero()) && (point2->y == g2_point_field_t::zero()) &&
           (point2->z == g2_point_field_t::zero()));
}
#endif