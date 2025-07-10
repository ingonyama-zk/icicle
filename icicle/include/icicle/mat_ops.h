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
    icicleStreamHandle stream = nullptr; ///< stream for async execution.
    bool is_a_on_device = false;         ///< True if `a` resides on device memory.
    bool is_b_on_device = false;         ///< True if `b` resides on device memory.
    bool is_result_on_device = false;    ///< True to keep result on device, else host.
    bool a_transposed = false;           ///< True if `a` is pre-transposed.
    bool b_transposed = false;           ///< True if `b` is pre-transposed.
    bool result_transposed = false;      ///< True to transpose the output.
    bool is_async = false;               ///< True for non-blocking call; user must sync stream.
    ConfigExtension* ext = nullptr;      ///< Optional backend-specific settings.
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
