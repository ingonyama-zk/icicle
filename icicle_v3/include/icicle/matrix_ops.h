#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"
#include "icicle/vec_ops.h"

using namespace field_config;

namespace icicle {
  // TODO Yuval: move to vec_ops like in V2
  /*************************** Frontend APIs ***************************/

  /**
   * Transpose a matrix out-of-place.
   * @param mat_in array of type E with size rows * cols.
   * @param nof_rows number of rows.
   * @param nof_cols number of columns.
   * @param config Configuration of the operation.
   * @param arr_out buffer of the same size as `mat_in` to write the transpose matrix into.
   * @tparam E The type of elements `mat_in' and `mat_out`.
   * @return `SUCCESS` if the execution was successful and an error code otherwise.
   */
  template <typename E>
  eIcicleError
  matrix_transpose(const E* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, E* mat_out);

  // field specific APIs. TODO Yuval move to api headers like icicle V2
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, matrix_transpose)(
    const scalar_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, scalar_t* mat_out);

  /*************************** Backend registration ***************************/

  using scalarMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    scalar_t* out)>;

  void register_matrix_transpose(const std::string& deviceType, scalarMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose) = []() -> bool {                                                         \
      register_matrix_transpose(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef EXT_FIELD
  using extFieldMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    extension_t* out)>;

  void register_extension_matrix_transpose(const std::string& deviceType, extFieldMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose_ext_field) = []() -> bool {                                               \
      register_extension_matrix_transpose(DEVICE_TYPE, FUNC);                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // EXT_FIELD

} // namespace icicle