#include "../curves/curve_config.cuh"
#include "device_context.cuh"
#include "mont.cuh"

namespace mont {
  extern "C" cudaError_t
  ScalarConvertMontgomery(curve_config::scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t
  AffineConvertMontgomery(curve_config::affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t ProjectiveConvertMontgomery(
    curve_config::projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }
} // namespace mont