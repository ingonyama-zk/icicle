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
   *  - `A` is the [affine representation](@ref affine_t) of curve points;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, precompute_msm_bases_cuda)(
    affine_t* bases,
    int msm_size,
    MSMConfig& config,
    affine_t* output_bases)
  {
    return precompute_msm_bases<affine_t, projective_t>(
      bases, msm_size, config, output_bases);
  }

  /**
   * Extern "C" version of [msm](@ref msm) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   *  - `A` is the [affine representation](@ref affine_t) of curve points;
   *  - `P` is the [projective representation](@ref projective_t) of curve points.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, msm_cuda)(
    const scalar_t* scalars, const affine_t* points, int msm_size, MSMConfig& config, projective_t* out)
  {
    return msm<scalar_t, affine_t, projective_t>(scalars, points, msm_size, config, out);
  }
} // namespace msm