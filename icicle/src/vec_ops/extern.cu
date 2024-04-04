#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/utils.h"
#include "vec_ops.cu"

namespace vec_ops {
  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, MulCuda)(
    scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig<scalar_t>& config, scalar_t* result)
  {
    return Mul<scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, AddCuda)(
    scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig<scalar_t>& config, scalar_t* result)
  {
    return Add<scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, SubCuda)(
    scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig<scalar_t>& config, scalar_t* result)
  {
    return Sub<scalar_t>(vec_a, vec_b, n, config, result);
  }
} // namespace vec_ops