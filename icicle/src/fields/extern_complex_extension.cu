#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/mont.cuh"
#include "utils/utils.h"
#include "gpu-utils/device_context.cuh"

extern "C" void CONCAT_EXPAND(FIELD, c_extension_generate_scalars)(c_extension_t* scalars, int size)
{
  c_extension_t::rand_host_many(scalars, size);
}

extern "C" cudaError_t CONCAT_EXPAND(FIELD, c_extension_scalar_convert_montgomery)(
  c_extension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}
