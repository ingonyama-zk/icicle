#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/utils.h"
#include "vec_ops.cu"

namespace vec_ops {
  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, mul_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return mul<scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, add_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return add<scalar_t>(vec_a, vec_b, n, config, result);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_cuda)(
    scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return fold<scalar_t, scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Accumulate (as vec_a[i] += vec_b[i]) function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, accumulate_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config)
  {
    return add<scalar_t>(vec_a, vec_b, n, config, vec_a);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, stwo_convert_cuda)(
    uint32_t* vec_a, uint32_t* vec_b, uint32_t* vec_c, uint32_t* vec_d, int n, scalar_t* result)
  {
    return stwo_convert<scalar_t>(vec_a, vec_b, vec_c, vec_d, n, result, true);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, sub_cuda)(scalar_t* vec_a, scalar_t* vec_b, int n, VecOpsConfig& config, scalar_t* result)
  {
    return sub<scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of transpose_batch function with the template parameter
   * `E` being the [field](@ref scalar_t) (either scalar field of the curve given by `-DCURVE`
   * or standalone "STARK field" given by `-DFIELD`).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, transpose_matrix_cuda)(
    const scalar_t* input,
    uint32_t row_size,
    uint32_t column_size,
    scalar_t* output,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    return transpose_matrix<scalar_t>(input, output, row_size, column_size, ctx, on_device, is_async);
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, bit_reverse_cuda)(const scalar_t* input, uint64_t n, BitReverseConfig& config, scalar_t* output)
  {
    return bit_reverse<scalar_t>(input, n, config, output);
  }
} // namespace vec_ops