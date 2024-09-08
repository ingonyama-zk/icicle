#include "fields/field_config.cuh"

using namespace field_config;

#include "utils/utils.h"
#include "vec_ops.cu"

namespace vec_ops {
  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [extension field](@ref cextension_t) of the base field given by `-DFIELD` env variable
   * during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_mul_cuda)(
    cextension_t* vec_a, cextension_t* vec_b, int n, VecOpsConfig& config, cextension_t* result)
  {
    return mul<cextension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [extension field](@ref cextension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_add_cuda)(
    cextension_t* vec_a, cextension_t* vec_b, int n, VecOpsConfig& config, cextension_t* result)
  {
    return add<cextension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   *  Accumulate (as vec_a[i] += vec_b[i]) function with the template parameter
   * `E` being the [extension field](@ref cextension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, cextension_accumulate_cuda)(cextension_t* vec_a, cextension_t* vec_b, int n, VecOpsConfig& config)
  {
    return add<cextension_t>(vec_a, vec_b, n, config, vec_a);
  }

    extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_stwo_convert_cuda)(
    uint32_t* vec_a, uint32_t* vec_b, uint32_t* vec_c, uint32_t* vec_d, int n, cextension_t* result)
  {
    return  stwo_convert<cextension_t>(vec_a, vec_b, vec_c, vec_d, n, result, true);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [extension field](@ref cextension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_sub_cuda)(
    cextension_t* vec_a, cextension_t* vec_b, int n, VecOpsConfig& config, cextension_t* result)
  {
    return sub<cextension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of transpose_batch function with the template parameter
   * `E` being the [extension field](@ref cextension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_transpose_matrix_cuda)(
    const cextension_t* input,
    uint32_t row_size,
    uint32_t column_size,
    cextension_t* output,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    return transpose_matrix<cextension_t>(input, output, row_size, column_size, ctx, on_device, is_async);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, cextension_bit_reverse_cuda)(
    const cextension_t* input, uint64_t n, BitReverseConfig& config, cextension_t* output)
  {
    return bit_reverse<cextension_t>(input, n, config, output);
  }
} // namespace vec_ops