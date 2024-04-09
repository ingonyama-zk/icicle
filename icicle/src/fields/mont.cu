#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/mont.cuh"
#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace mont {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ScalarConvertMontgomery)(
    scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }
} // namespace mont