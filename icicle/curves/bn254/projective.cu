#include <cuda.h>
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" BN254::projective_t random_projective_bn254()
{
  return BN254::projective_t::rand_host();
}

extern "C" BN254::projective_t projective_zero_bn254()
{
  return BN254::projective_t::zero();
}

extern "C" bool projective_is_on_curve_bn254(BN254::projective_t *point1)
{
  return BN254::projective_t::is_on_curve(*point1);
}

extern "C" BN254::affine_t projective_to_affine_bn254(BN254::projective_t *point1)
{
  return BN254::projective_t::to_affine(*point1);
}

extern "C" BN254::projective_t projective_from_affine_bn254(BN254::affine_t *point1)
{
  return BN254::projective_t::from_affine(*point1);
}

extern "C" BN254::scalar_field_t random_scalar_bn254()
{
  return BN254::scalar_field_t::rand_host();
}

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

extern "C" BN254::g2_projective_t random_g2_projective_bn254()
{
  return BN254::g2_projective_t::rand_host();
}

extern "C" BN254::g2_affine_t g2_projective_to_affine_bn254(BN254::g2_projective_t *point1)
{
  return BN254::g2_projective_t::to_affine(*point1);
}

extern "C" BN254::g2_projective_t g2_projective_from_affine_bn254(BN254::g2_affine_t *point1)
{
  return BN254::g2_projective_t::from_affine(*point1);
}

extern "C" bool g2_projective_is_on_curve_bn254(BN254::g2_projective_t *point1)
{
  return BN254::g2_projective_t::is_on_curve(*point1);
}

#endif
