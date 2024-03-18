#include "curves/curve_config.cuh"
#include "device_context.cuh"
#include "mont.cuh"
#include "utils/utils.h"

namespace mont {
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ScalarConvertMontgomery)(
    curve_config::scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, AffineConvertMontgomery)(
    curve_config::affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ProjectiveConvertMontgomery)(
    curve_config::projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

#if defined(G2_DEFINED)

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2AffineConvertMontgomery)(
    curve_config::g2_affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2ProjectiveConvertMontgomery)(
    curve_config::g2_projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

#endif
} // namespace mont