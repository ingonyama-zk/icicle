#include "fields/field_config.cuh"

using namespace field_config;

#include "ntt.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace ntt {
  /**
   * Extern "C" version of [InitDomain](@ref InitDomain) function with the following
   * value of template parameter (where the curve is given by `-DFIELD` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, InitializeDomain)(
    scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    return InitDomain(*primitive_root, ctx, fast_twiddles_mode);
  }

  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the curve is given by `-DFIELD` env variable during build):
   *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, NTTCuda)(
    const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return NTT<scalar_t, scalar_t>(input, size, dir, config, output);
  }
} // namespace ntt