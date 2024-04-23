#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/mont.cuh"
#include "utils/utils.h"
#include "gpu-utils/device_context.cuh"

extern "C" void CONCAT_EXPAND(FIELD, generate_scalars)(scalar_t* scalars, int size)
{
  scalar_t::rand_host_many(scalars, size);
}

extern "C" cudaError_t CONCAT_EXPAND(FIELD, scalar_convert_montgomery)(
  scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}
