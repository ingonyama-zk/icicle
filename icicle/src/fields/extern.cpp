#define FIELD_ID BN254 
#include "../../include/fields/field_config.cuh"

using namespace field_config;

//#include "../../include/utils/mont.cuh"
#include "../../include/utils/utils.h"
#include "../../include/gpu-utils/device_context.cuh"

extern "C" void CONCAT_EXPAND(FIELD, generate_scalars)(scalar_t* scalars, int size)
{
  return;
}

extern "C" int CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(
  scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  return 0;
}
