#include "fields/field_config.cuh"

using namespace field_config;

#include "ntt.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace ntt {
  /**
   * Extern "C" version of [init_domain](@ref init_domain) function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, initialize_domain)(
    scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    return init_domain(*primitive_root, ctx, fast_twiddles_mode);
  }

  /**
   * Extern "C" version of [ntt](@ref ntt) function with the following values of template parameters
   * (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ntt_cuda)(
    const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return ntt<scalar_t, scalar_t>(input, size, dir, config, output);
  }

  /**
   * Extern "C" version of [release_domain](@ref release_domain) function with the following values of template parameters
   * (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, release_domain)(device_context::DeviceContext& ctx)
  {
    return release_domain<scalar_t>(ctx);
  }

  /**
   * Extern "C" version of [GetRootOfUnity](@ref GetRootOfUnity) function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   */
  extern "C" scalar_t CONCAT_EXPAND(FIELD, get_root_of_unity)(uint32_t logn) { return get_root_of_unity<scalar_t>(logn); }
} // namespace ntt