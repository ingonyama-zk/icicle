#define CURVE_ID BN254
#include "../../include/curves/curve_config.cuh"

using namespace curve_config;

#include "../../include/gpu-utils/device_context.cuh"
#include "../../include/utils/utils.h"
// #include "../utils/mont.cuh"

extern "C" bool CONCAT_EXPAND(CURVE, eq)(projective_t* point1, projective_t* point2)
{
  return true;
}

extern "C" void CONCAT_EXPAND(CURVE, to_affine)(projective_t* point, affine_t* point_out)
{
  return;
}

extern "C" void CONCAT_EXPAND(CURVE, generate_projective_points)(projective_t* points, int size)
{
  return;
}

extern "C" void CONCAT_EXPAND(CURVE, generate_affine_points)(affine_t* points, int size)
{
  return;
}

extern "C" int CONCAT_EXPAND(CURVE, affine_convert_montgomery)(
  affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  return 0;
}

extern "C" int CONCAT_EXPAND(CURVE, projective_convert_montgomery)(
  projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  return 0;
}
