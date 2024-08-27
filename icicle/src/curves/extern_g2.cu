#include "curves/curve_config.cuh"

using namespace curve_config;

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"
#include "utils/mont.cuh"

extern "C" bool CONCAT_EXPAND(CURVE, g2_eq)(g2_projective_t* point1, g2_projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == g2_point_field_t::zero()) && (point1->y == g2_point_field_t::zero()) &&
           (point1->z == g2_point_field_t::zero())) &&
         !((point2->x == g2_point_field_t::zero()) && (point2->y == g2_point_field_t::zero()) &&
           (point2->z == g2_point_field_t::zero()));
}

extern "C" void CONCAT_EXPAND(CURVE, g2_add)(g2_projective_t* point1, g2_projective_t* point2, g2_projective_t* result)
{
  *result = *point1 + *point2;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_mul_scalar)(g2_projective_t* point1, scalar_t* scalar, g2_projective_t* result)
{
  *result = *point1 * *scalar;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_to_affine)(g2_projective_t* point, g2_affine_t* point_out)
{
  *point_out = g2_projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_projective_points)(g2_projective_t* points, int size)
{
  g2_projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_affine_points)(g2_affine_t* points, int size)
{
  g2_projective_t::rand_host_many_affine(points, size);
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, g2_affine_convert_montgomery)(
  g2_affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, g2_projective_convert_montgomery)(
  g2_projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}