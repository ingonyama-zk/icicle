#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/utils.h"
#include "vec_ops.cu"

namespace vec_ops {
  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable
   * during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_mul_cuda)(
    extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config, extension_t* result)
  {
    return mul<extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_add_cuda)(
    extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config, extension_t* result)
  {
    return add<extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   *  Accumulate (as vec_a[i] += vec_b[i]) function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, q_extension_accumulate_cuda)(extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config)
  {
    return add<extension_t>(vec_a, vec_b, n, config, vec_a);
  }

    extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_stwo_convert_cuda)(
    uint32_t* vec_a, uint32_t* vec_b, uint32_t* vec_c, uint32_t* vec_d, int n, extension_t* result)
  {
    return  stwo_convert<extension_t>(vec_a, vec_b, vec_c, vec_d, n, result, true);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_sub_cuda)(
    extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config, extension_t* result)
  {
    return sub<extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of transpose_batch function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, extension_transpose_matrix_cuda)(
    const extension_t* input,
    uint32_t row_size,
    uint32_t column_size,
    extension_t* output,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    return transpose_matrix<extension_t>(input, output, row_size, column_size, ctx, on_device, is_async);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, q_extension_bit_reverse_cuda)(
    const extension_t* input, uint64_t n, BitReverseConfig& config, extension_t* output)
  {
    return bit_reverse<extension_t>(input, n, config, output);
  }
} // namespace vec_ops