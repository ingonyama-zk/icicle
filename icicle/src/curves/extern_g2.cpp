#include "curves/curve_config.cuh"

using namespace curve_config;

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"
#include "utils/mont.cuh"

extern "C" bool CONCAT_EXPAND(CURVE, g2_eq)(g2_projective_t* point1, g2_projective_t* point2)
{
  return true;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_to_affine)(g2_projective_t* point, g2_affine_t* point_out)
{
  return;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_projective_points)(g2_projective_t* points, int size)
{
  return;
}

extern "C" void CONCAT_EXPAND(CURVE, g2_generate_affine_points)(g2_affine_t* points, int size)
{
  return;
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, g2_affine_convert_montgomery)(
  g2_affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  return 0;
}

extern "C" cudaError_t CONCAT_EXPAND(CURVE, g2_projective_convert_montgomery)(
  g2_projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  return 0;
}