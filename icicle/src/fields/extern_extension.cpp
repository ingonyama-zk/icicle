#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/mont.cuh"
#include "utils/utils.h"
#include "gpu-utils/device_context.cuh"

extern "C" void CONCAT_EXPAND(FIELD, extension_generate_scalars)(extension_t* scalars, int size)
{
  return;
}

extern "C" cudaError_t CONCAT_EXPAND(FIELD, extension_scalar_convert_montgomery)(
  extension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  return 0;
}
