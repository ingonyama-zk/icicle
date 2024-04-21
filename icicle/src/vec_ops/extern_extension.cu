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
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ExtensionMulCuda)(
    extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config, extension_t* result)
  {
    return Mul<extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ExtensionAddCuda)(
    extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config, extension_t* result)
  {
    return Add<extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ExtensionSubCuda)(
    extension_t* vec_a, extension_t* vec_b, int n, VecOpsConfig& config, extension_t* result)
  {
    return Sub<extension_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of transpose_batch function with the template parameter
   * `E` being the [extension field](@ref extension_t) of the base field given by `-DFIELD` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, ExtensionTransposeMatrix)(
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
} // namespace vec_ops