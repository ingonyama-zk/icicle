#include "gpu-utils/device_context.cuh"
#include "utils/mont.cuh"
#include "utils/utils.h"

namespace mont {
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, AffineConvertMontgomery)(
    affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ProjectiveConvertMontgomery)(
    projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

#if defined(G2_DEFINED)

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2AffineConvertMontgomery)(
    g2_affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2ProjectiveConvertMontgomery)(
    g2_projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)
  {
    if (is_into) {
      return ToMontgomery(d_inout, n, ctx.stream, d_inout);
    } else {
      return FromMontgomery(d_inout, n, ctx.stream, d_inout);
    }
  }

#endif
} // namespace mont