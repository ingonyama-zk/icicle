#pragma once

#include <functional>

#include "errors.h"
#include "runtime.h"

#include "icicle/fields/field.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"
#include "icicle/program/program.h"

namespace icicle {

  /*************************** Frontend APIs ***************************/
  /**
   * @brief Configuration for matrix operations.
   * @note APIs with a single input, ignore input b.
   */
  struct MatMulConfig {
    icicleStreamHandle stream = nullptr; /** Stream for asynchronous execution. */
    bool is_a_on_device = false;         /** True if `a` is on the device, false if it is not. Default value: false. */
    bool is_b_on_device =
      false; /** True if `b` is on the device, false if it is not. Default value: false. OPTIONAL. */
    bool is_result_on_device = false; /** If true, the output is preserved on the device, otherwise on the host. Default
                                    value: false. */
    bool is_async = false;            /** Whether to run the matrix operations asynchronously.
                                    If set to `true`, the function will be non-blocking and synchronization
                                    must be explicitly managed using `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                    If set to `false`, the function will block the current CPU thread. */
    int batch_size = 1;               /** Number of matrices (or operations) to process in a batch.
                                        Each matrix operation will be performed independently on each batch element.
                                        Default value: 1. */
    bool columns_batch = false;       /** True if the batched matrices are stored as separate matrices in a 3D array.
                                   If false, the batched matrices are stored contiguously in memory.
                                   Default value: false. */
    bool a_transposed = false;      /** True if the matrix a is transposed, false if it is not. Default value: false. */
    bool b_transposed = false;      /** True if the matrix b is transposed, false if it is not. Default value: false. */
    bool result_transposed = false; /** True if the result is transposed, false if it is not. Default value: false. */
    ConfigExtension* ext = nullptr; /** Backend-specific extension. */
  };

  /**
   * @brief Returns the default value of MatOpsConfig.
   *
   * @return Default value of MatOpsConfig.
   */
  static MatMulConfig default_mat_mul_config() { return MatMulConfig{}; }

  /**
   * @brief Multiplies two matrices.
   *
   * @tparam T Type of the elements in the matrices.
   * @param mat_a Pointer to the first input matrix.
   * @param nof_rows_a Number of rows in the first input matrix.
   * @param nof_cols_a Number of columns in the first input matrix.
   * @param mat_b Pointer to the second input matrix.
   * @param nof_rows_b Number of rows in the second input matrix.
   * @param nof_cols_b Number of columns in the second input matrix.
   * @param config Configuration for the operation.
   * @param mat_out Pointer to the output matrix where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   * @note The input matrices are assumed to be stored in row-major order.
   *       This function multiplies an matrix A  or a batch of matrices with matrix B.
   */
  template <typename T>
  eIcicleError matmul(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out);

} // namespace icicle
