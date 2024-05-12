
#define FIELD_ID BN254
#define CURVE_ID BN254
#include "../../include/curves/curve_config.cuh"
#include "../../include/fields/field_config.cuh"

using namespace curve_config;
using namespace field_config;

#include "ntt.cpp"

#include "../../include/gpu-utils/device_context.cuh"
#include "../../include/utils/utils.h"

namespace ntt {
  /**
   * Extern "C" version of [ntt](@ref ntt) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [projective representation](@ref projective_t) of the curve (i.e. EC NTT is computed);
   *  - `E` is the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ecntt_cuda)(
    const projective_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, projective_t* output)
  {
    return ntt<scalar_t, projective_t>(input, size, dir, config, output);
  }
} // namespace ntt
