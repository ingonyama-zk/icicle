#include "fields/field_config.cuh"

using namespace field_config;

#include "ntt.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace ntt {
  /**
   * Extern "C" version of [ntt](@ref ntt) function with the following values of template parameters
   * (where the field is given by `-DFIELD` env variable during build):
   *  - `E` is the [field](@ref scalar_t);
   *  - `S` is the [extension](@ref extension_t) of `E` of appropriate degree;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, extension_ntt_cuda)(
    const extension_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, extension_t* output)
  {
    return ntt<scalar_t, extension_t>(input, size, dir, config, output);
  }
} // namespace ntt
