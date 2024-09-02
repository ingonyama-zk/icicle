#include "curves/curve_config.cuh"

using namespace curve_config;

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"
#include "utils/mont.cuh"

extern "C" bool CONCAT_EXPAND(CURVE, eq)(projective_t* point1, projective_t* point2)
{
  return (*point1 == *point2) &&
         !((point1->x == point_field_t::zero()) && (point1->y == point_field_t::zero()) &&
           (point1->z == point_field_t::zero())) &&
         !((point2->x == point_field_t::zero()) && (point2->y == point_field_t::zero()) &&
           (point2->z == point_field_t::zero()));
}

extern "C" void CONCAT_EXPAND(CURVE, sub)(projective_t* point1, projective_t* point2, projective_t* result)
{
  *result = *point1 - *point2;
}

extern "C" void CONCAT_EXPAND(CURVE, add)(projective_t* point1, projective_t* point2, projective_t* result)
{
  *result = *point1 + *point2;
}

extern "C" void CONCAT_EXPAND(CURVE, mul_scalar)(projective_t* point1, scalar_t* scalar, projective_t* result)
{
  *result = *point1 * *scalar;
}

extern "C" void CONCAT_EXPAND(CURVE, mul_two_scalar)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 * *scalar2;
}

extern "C" void CONCAT_EXPAND(CURVE, to_affine)(projective_t* point, affine_t* point_out)
{
  *point_out = projective_t::to_affine(*point);
}

extern "C" void CONCAT_EXPAND(CURVE, generate_projective_points)(projective_t* points, int size)
{
  projective_t::rand_host_many(points, size);
}

extern "C" void CONCAT_EXPAND(CURVE, generate_affine_points)(affine_t* points, int size)
{
  projective_t::rand_host_many_affine(points, size);
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, affine_convert_montgomery)(
  affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, projective_convert_montgomery)(
  projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}
