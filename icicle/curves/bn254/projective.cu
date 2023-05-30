#include <cuda.h>
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" bool eq_bn254(BN254::projective_t *point1, BN254::projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BN254::point_field_t::zero()) && (point1->y == BN254::point_field_t::zero()) && (point1->z == BN254::point_field_t::zero())) && 
  !((point2->x == BN254::point_field_t::zero()) && (point2->y == BN254::point_field_t::zero()) && (point2->z == BN254::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_bn254(BN254::g2_projective_t *point1, BN254::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BN254::g2_point_field_t::zero()) && (point1->y == BN254::g2_point_field_t::zero()) && (point1->z == BN254::g2_point_field_t::zero())) && 
  !((point2->x == BN254::g2_point_field_t::zero()) && (point2->y == BN254::g2_point_field_t::zero()) && (point2->z == BN254::g2_point_field_t::zero()));
}
#endif