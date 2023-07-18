#include <cuda.h>
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" bool eq_bls12_381(BLS12_381::projective_t *point1, BLS12_381::projective_t *point2)
{
    return (*point1 == *point2) && 
    !((point1->x == BLS12_381::point_field_t::zero()) && (point1->y == BLS12_381::point_field_t::zero()) && (point1->z == BLS12_381::point_field_t::zero())) && 
    !((point2->x == BLS12_381::point_field_t::zero()) && (point2->y == BLS12_381::point_field_t::zero()) && (point2->z == BLS12_381::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_bls12_381(BLS12_381::g2_projective_t *point1, BLS12_381::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BLS12_381::g2_point_field_t::zero()) && (point1->y == BLS12_381::g2_point_field_t::zero()) && (point1->z == BLS12_381::g2_point_field_t::zero())) && 
  !((point2->x == BLS12_381::g2_point_field_t::zero()) && (point2->y == BLS12_381::g2_point_field_t::zero()) && (point2->z == BLS12_381::g2_point_field_t::zero()));
}
#endif
