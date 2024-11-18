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
#ifdef DCCT
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, initialize_domain)(uint32_t logn, c_extension_t* primitive_root, device_context::DeviceContext& ctx)
  {
    return init_domain<scalar_t, c_extension_t>(logn, *primitive_root, ctx);
  }
#else
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, initialize_domain)(uint32_t logn, 
    scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    return init_domain(logn, *primitive_root, ctx, fast_twiddles_mode);
  }
#endif

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
   * Extern "C" version of [release_domain](@ref release_domain) function with the following values of template
   * parameters (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, release_domain)(uint32_t logn, device_context::DeviceContext& ctx)
  {
    return release_domain<scalar_t>(logn, ctx);
  }

  /**
   * Extern "C" version of [get_root_of_unity](@ref get_root_of_unity) function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   */
#ifdef DCCT
  extern "C" void CONCAT_EXPAND(FIELD, get_root_of_unity)(uint32_t logn, c_extension_t* output)
  {
    *output = get_root_of_unity<scalar_t, c_extension_t>(logn);
  }
#else
  extern "C" void CONCAT_EXPAND(FIELD, get_root_of_unity)(uint32_t logn, scalar_t* output)
  {
    *output = get_root_of_unity<scalar_t>(logn);
  }
#endif
} // namespace ntt