#include <cuda.h>
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" bool eq_${CURVE_NAME_L}(${CURVE_NAME_U}::projective_t *point1, ${CURVE_NAME_U}::projective_t *point2)
{
    return (*point1 == *point2) && 
    !((point1->x == ${CURVE_NAME_U}::point_field_t::zero()) && (point1->y == ${CURVE_NAME_U}::point_field_t::zero()) && (point1->z == ${CURVE_NAME_U}::point_field_t::zero())) && 
    !((point2->x == ${CURVE_NAME_U}::point_field_t::zero()) && (point2->y == ${CURVE_NAME_U}::point_field_t::zero()) && (point2->z == ${CURVE_NAME_U}::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_${CURVE_NAME_L}(${CURVE_NAME_U}::g2_projective_t *point1, ${CURVE_NAME_U}::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == ${CURVE_NAME_U}::g2_point_field_t::zero()) && (point1->y == ${CURVE_NAME_U}::g2_point_field_t::zero()) && (point1->z == ${CURVE_NAME_U}::g2_point_field_t::zero())) && 
  !((point2->x == ${CURVE_NAME_U}::g2_point_field_t::zero()) && (point2->y == ${CURVE_NAME_U}::g2_point_field_t::zero()) && (point2->z == ${CURVE_NAME_U}::g2_point_field_t::zero()));
}
#endif