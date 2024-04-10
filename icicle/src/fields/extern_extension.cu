#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/mont.cuh"
#include "utils/utils.h"
#include "gpu-utils/device_context.cuh"

extern "C" void CONCAT_EXPAND(FIELD, GenerateExtScalars)(extension_t* scalars, int size)
{
  extension_t::RandHostMany(scalars, size);
}

extern "C" cudaError_t CONCAT_EXPAND(FIELD, ExtScalarConvertMontgomery)(
  extension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
{
  if (is_into) {
    return mont::ToMontgomery(d_inout, n, ctx.stream, d_inout);
  } else {
    return mont::FromMontgomery(d_inout, n, ctx.stream, d_inout);
  }
}
