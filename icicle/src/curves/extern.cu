#include "curves/curve_config.cuh"

using namespace curve_config;

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"
#include "utils/mont.cuh"

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

extern "C" cudaError_t CONCAT_EXPAND(CURVE, AffineConvertMontgomery)(
  affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::ToMontgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::FromMontgomery(d_inout, n, ctx.stream, d_inout);
  }
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, ProjectiveConvertMontgomery)(
  projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::ToMontgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::FromMontgomery(d_inout, n, ctx.stream, d_inout);
  }
}
