
#include <cuda.h>

#include "curve_config.cuh"

#include "../../primitives/projective.cuh"

extern "C" bool eq_bls12_377(BLS12_377::projective_t *point1, BLS12_377::projective_t *point2)
{
    return (*point1 == *point2) && 
    !((point1->x == BLS12_377::point_field_t::zero()) && (point1->y == BLS12_377::point_field_t::zero()) && (point1->z == BLS12_377::point_field_t::zero())) && 
    !((point2->x == BLS12_377::point_field_t::zero()) && (point2->y == BLS12_377::point_field_t::zero()) && (point2->z == BLS12_377::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_bls12_377(BLS12_377::g2_projective_t *point1, BLS12_377::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BLS12_377::g2_point_field_t::zero()) && (point1->y == BLS12_377::g2_point_field_t::zero()) && (point1->z == BLS12_377::g2_point_field_t::zero())) && 
  !((point2->x == BLS12_377::g2_point_field_t::zero()) && (point2->y == BLS12_377::g2_point_field_t::zero()) && (point2->z == BLS12_377::g2_point_field_t::zero()));
}
#endif
