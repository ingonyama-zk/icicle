#include <cuda.h>
#include "../curves/bls12_381/curve_config.cuh"
#include "../curves/bls12_377/curve_config.cuh"
#include "../curves/bn254/curve_config.cuh"
#include "projective.cuh"

extern "C" bool eq_bls12_381(BLS12_381::projective_t *point1, BLS12_381::projective_t *point2)
{
    return (*point1 == *point2) && 
    !((point1->x == BLS12_381::point_field_t::zero()) && (point1->y == BLS12_381::point_field_t::zero()) && (point1->z == BLS12_381::point_field_t::zero())) && 
    !((point2->x == BLS12_381::point_field_t::zero()) && (point2->y == BLS12_381::point_field_t::zero()) && (point2->z == BLS12_381::point_field_t::zero()));
}

extern "C" bool eq_bls12_377(BLS12_377::projective_t *point1, BLS12_377::projective_t *point2)
{
    return (*point1 == *point2) && 
    !((point1->x == BLS12_377::point_field_t::zero()) && (point1->y == BLS12_377::point_field_t::zero()) && (point1->z == BLS12_377::point_field_t::zero())) && 
    !((point2->x == BLS12_377::point_field_t::zero()) && (point2->y == BLS12_377::point_field_t::zero()) && (point2->z == BLS12_377::point_field_t::zero()));
}

extern "C" bool eq_bn254(BN254::projective_t *point1, BN254::projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BN254::point_field_t::zero()) && (point1->y == BN254::point_field_t::zero()) && (point1->z == BN254::point_field_t::zero())) && 
  !((point2->x == BN254::point_field_t::zero()) && (point2->y == BN254::point_field_t::zero()) && (point2->z == BN254::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_bls12_381(BLS12_381::g2_projective_t *point1, BLS12_381::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BLS12_381::g2_point_field_t::zero()) && (point1->y == BLS12_381::g2_point_field_t::zero()) && (point1->z == BLS12_381::g2_point_field_t::zero())) && 
  !((point2->x == BLS12_381::g2_point_field_t::zero()) && (point2->y == BLS12_381::g2_point_field_t::zero()) && (point2->z == BLS12_381::g2_point_field_t::zero()));
}

extern "C" bool eq_g2_bls12_377(BLS12_377::g2_projective_t *point1, BLS12_377::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BLS12_377::g2_point_field_t::zero()) && (point1->y == BLS12_377::g2_point_field_t::zero()) && (point1->z == BLS12_377::g2_point_field_t::zero())) && 
  !((point2->x == BLS12_377::g2_point_field_t::zero()) && (point2->y == BLS12_377::g2_point_field_t::zero()) && (point2->z == BLS12_377::g2_point_field_t::zero()));
}

extern "C" bool eq_g2_bn254(BN254::g2_projective_t *point1, BN254::g2_projective_t *point2)
{
  return (*point1 == *point2) && 
  !((point1->x == BN254::g2_point_field_t::zero()) && (point1->y == BN254::g2_point_field_t::zero()) && (point1->z == BN254::g2_point_field_t::zero())) && 
  !((point2->x == BN254::g2_point_field_t::zero()) && (point2->y == BN254::g2_point_field_t::zero()) && (point2->z == BN254::g2_point_field_t::zero()));
}
#endif