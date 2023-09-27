#include "../../primitives/projective.cuh"
#include "curve_config.cuh"
#include <cuda.h>

extern "C" int random_projective_bw6_761(BW6_761::projective_t* out) { 
  try {
    out[0] = BW6_761::projective_t::rand_host();
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" BW6_761::projective_t projective_zero_bw6_761() { return BW6_761::projective_t::zero(); }

extern "C" bool projective_is_on_curve_bw6_761(BW6_761::projective_t* point1)
{
  return BW6_761::projective_t::is_on_curve(*point1);
}

extern "C" int projective_to_affine_bw6_761(BW6_761::affine_t* out, BW6_761::projective_t* point1)
{
  try {
    out[0] = BW6_761::projective_t::to_affine(*point1);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int projective_from_affine_bw6_761(BW6_761::projective_t* out, BW6_761::affine_t* point1)
{
  try {
    out[0] = BW6_761::projective_t::from_affine(*point1);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int random_scalar_bw6_761(BW6_761::scalar_t* out) { 
  try {
    out[0] = BW6_761::scalar_t::rand_host();
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" bool eq_bw6_761(BW6_761::projective_t* point1, BW6_761::projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == BW6_761::point_field_t::zero()) && (point1->y == BW6_761::point_field_t::zero()) &&
           (point1->z == BW6_761::point_field_t::zero())) &&
         !((point2->x == BW6_761::point_field_t::zero()) && (point2->y == BW6_761::point_field_t::zero()) &&
           (point2->z == BW6_761::point_field_t::zero()));
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2_bw6_761(BW6_761::g2_projective_t* point1, BW6_761::g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == BW6_761::point_field_t::zero()) && (point1->y == BW6_761::point_field_t::zero()) &&
           (point1->z == BW6_761::point_field_t::zero())) &&
         !((point2->x == BW6_761::point_field_t::zero()) && (point2->y == BW6_761::point_field_t::zero()) &&
           (point2->z == BW6_761::point_field_t::zero()));
}

extern "C" int random_g2_projective_bw6_761(BW6_761::g2_projective_t* out) 
{ 
  try {
    out[0] = BW6_761::g2_projective_t::rand_host();
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int g2_projective_to_affine_bw6_761(BW6_761::g2_affine_t* out, BW6_761::g2_projective_t* point1)
{
  try {
    out[0] = BW6_761::g2_projective_t::to_affine(*point1);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int g2_projective_from_affine_bw6_761(BW6_761::g2_projective_t* out, BW6_761::g2_affine_t* point1)
{
  try {
    out[0] = BW6_761::g2_projective_t::from_affine(*point1);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" bool g2_projective_is_on_curve_bw6_761(BW6_761::g2_projective_t* point1)
{
  return BW6_761::g2_projective_t::is_on_curve(*point1);
}
#endif
