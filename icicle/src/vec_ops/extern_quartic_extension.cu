#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/utils.h"
#include "vec_ops.cu"

namespace vec_ops {
  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [extension field](@ref q_extension_t) of the base field given by `-DFIELD` env variable
   * during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_mul_cuda)(
    q_extension_t* vec_a, q_extension_t* vec_b, int n, VecOpsConfig& config, q_extension_t* result)
  {
    return mul<q_extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [extension field](@ref q_extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_add_cuda)(
    q_extension_t* vec_a, q_extension_t* vec_b, int n, VecOpsConfig& config, q_extension_t* result)
  {
    return add<q_extension_t>(vec_a, vec_b, n, config, result);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_fold_cuda)(
    scalar_t* vec_a, q_extension_t* vec_b, int n, VecOpsConfig& config, q_extension_t* result)
  {
    return fold<q_extension_t, scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   *  Accumulate (as vec_a[i] += vec_b[i]) function with the template parameter
   * `E` being the [extension field](@ref q_extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, q_extension_accumulate_cuda)(q_extension_t* vec_a, q_extension_t* vec_b, int n, VecOpsConfig& config)
  {
    return add<q_extension_t>(vec_a, vec_b, n, config, vec_a);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_stwo_convert_cuda)(
    uint32_t* vec_a, uint32_t* vec_b, uint32_t* vec_c, uint32_t* vec_d, int n, q_extension_t* result)
  {
    return  stwo_convert<q_extension_t>(vec_a, vec_b, vec_c, vec_d, n, result, true);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [extension field](@ref q_extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_sub_cuda)(
    q_extension_t* vec_a, q_extension_t* vec_b, int n, VecOpsConfig& config, q_extension_t* result)
  {
    return sub<q_extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of transpose_batch function with the template parameter
   * `E` being the [extension field](@ref q_extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, extension_transpose_matrix_cuda)(
    const q_extension_t* input,
    uint32_t row_size,
    uint32_t column_size,
    q_extension_t* output,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    return transpose_matrix<q_extension_t>(input, output, row_size, column_size, ctx, on_device, is_async);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_bit_reverse_cuda)(
    const q_extension_t* input, uint64_t n, BitReverseConfig& config, q_extension_t* output)
  {
    return bit_reverse<q_extension_t>(input, n, config, output);
  }
} // namespace vec_ops