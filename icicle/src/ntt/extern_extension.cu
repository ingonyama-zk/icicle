#include "fields/field_config.cuh"

using namespace field_config;

#include "ntt.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace ntt {
  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the field is given by `-DFIELD` env variable during build):
   *  - `E` is the [scalar field](@ref scalar_t) of the curve;
   *  - `S` is the [extension](@ref extension_t) of `E` of an appropriate degree;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ExtensionNTTCuda)(
    const extension_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, extension_t* output)
  {
    return NTT<scalar_t, extension_t>(input, size, dir, config, output);
  }
} // namespace ntt
