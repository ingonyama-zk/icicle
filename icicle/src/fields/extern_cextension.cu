#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/mont.cuh"
#include "utils/utils.h"
#include "gpu-utils/device_context.cuh"

extern "C" void CONCAT_EXPAND(FIELD, cextension_generate_scalars)(cextension_t* scalars, int size)
{
  cextension_t::rand_host_many(scalars, size);
}

extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_scalar_convert_montgomery)(
  cextension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::to_montgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::from_montgomery(d_inout, n, ctx.stream, d_inout);
  }
}
