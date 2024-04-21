#include "fields/field_config.cuh"

using namespace field_config;

#include "ntt.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace ntt {
  /**
   * Extern "C" version of [InitDomain](@ref InitDomain) function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, InitializeDomain)(
    scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    return InitDomain(*primitive_root, ctx, fast_twiddles_mode);
  }

  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, NTTCuda)(
    const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return NTT<scalar_t, scalar_t>(input, size, dir, config, output);
  }

  /**
   * Extern "C" version of [ReleaseDomain](@ref ReleaseDomain) function with the following values of template parameters
   * (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ReleaseDomain)(device_context::DeviceContext& ctx)
  {
    return ReleaseDomain<scalar_t>(ctx);
  }

  /**
   * Extern "C" version of [GetRootOfUnity](@ref GetRootOfUnity) function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   */
  extern "C" scalar_t CONCAT_EXPAND(FIELD, GetRootOfUnity)(uint32_t logn) { return GetRootOfUnity<scalar_t>(logn); }
} // namespace ntt