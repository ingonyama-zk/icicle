#include "utils/utils.h"
#include "curves/curve_config.cuh"

namespace ntt {
  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the curve is given by `-DFIELD` env variable during build):
   *  - `S` is the [projective representation](@ref projective_t) of the curve (i.e. EC NTT is computed);
   *  - `E` is the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ECNTTCuda)(
    curve_config::projective_t* input,
    int size,
    NTTDir dir,
    NTTConfig<curve_config::scalar_t>& config,
    curve_config::projective_t* output)
  {
    return NTT<curve_config::scalar_t, curve_config::projective_t>(input, size, dir, config, output);
  }
}