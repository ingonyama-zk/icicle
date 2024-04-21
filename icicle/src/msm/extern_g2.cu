#include "curves/curve_config.cuh"
#include "fields/field_config.cuh"

using namespace curve_config;
using namespace field_config;

#include "msm.cu"
#include "utils/utils.h"

namespace msm {
  /**
   * Extern "C" version of [precompute_msm_bases](@ref precompute_msm_bases) function with the following values of
   * template parameters (where the curve is given by `-DCURVE` env variable during build):
   *  - `A` is the [affine representation](@ref g2_affine_t) of G2 curve points;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, g2_precompute_msm_bases)(
    g2_affine_t* bases,
    int bases_size,
    int precompute_factor,
    int _c,
    bool are_bases_on_device,
    device_context::DeviceContext& ctx,
    g2_affine_t* output_bases)
  {
    return precompute_msm_bases<g2_affine_t, g2_projective_t>(
      bases, bases_size, precompute_factor, _c, are_bases_on_device, ctx, output_bases);
  }

  /**
   * Extern "C" version of [msm](@ref msm) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   *  - `A` is the [affine representation](@ref g2_affine_t) of G2 curve points;
   *  - `P` is the [projective representation](@ref g2_projective_t) of G2 curve points.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, g2_msm_cuda)(
    const scalar_t* scalars, const g2_affine_t* points, int msm_size, MSMConfig& config, g2_projective_t* out)
  {
    return msm<scalar_t, g2_affine_t, g2_projective_t>(scalars, points, msm_size, config, out);
  }
} // namespace msm