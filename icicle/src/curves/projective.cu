#include "curves/curve_config.cuh"
#include "fields/field_config.cuh"

using namespace curve_config;
using namespace field_config;

#include "curves/projective.cuh"
#include <cuda.h>
#include "utils/utils.h"

template <>
struct SharedMemory<projective_t> {
  __device__ projective_t* getPointer()
  {
    extern __shared__ projective_t s_projective_[];
    return s_projective_;
  }
};

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

#if defined(G2)

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
