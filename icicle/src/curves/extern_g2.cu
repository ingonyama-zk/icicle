#include "curves/curve_config.cuh"

using namespace curve_config;

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"
#include "utils/mont.cuh"

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

extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2AffineConvertMontgomery)(
  g2_affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::ToMontgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::FromMontgomery(d_inout, n, ctx.stream, d_inout);
  }
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2ProjectiveConvertMontgomery)(
  g2_projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::ToMontgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::FromMontgomery(d_inout, n, ctx.stream, d_inout);
  }
}