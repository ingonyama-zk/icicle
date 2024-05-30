#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

using namespace field_config;

namespace icicle {

  /*************************** Frontend APIs ***************************/

  /**
   * @struct MatrixOpsConfig
   * Struct that encodes parameters to be passed into matrix ops.
   */
  struct MatrixOpsConfig {
    icicleStreamHandle stream; /**< stream for async execution. */
    bool is_input_on_device;   /**< True if `a` is on device and false if it is not. Default value: false. */
    bool is_output_on_device;  /**< True if `b` is on device and false if it is not. Default value: false. */
    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */

    ConfigExtension ext; /** backend specific extensions*/
  };

  /**
   * A function that returns the default value of [MatrixOpsConfig](@ref MatrixOpsConfig).
   * @return Default value of [MatrixOpsConfig](@ref MatrixOpsConfig).
   */
  static MatrixOpsConfig default_matrix_ops_config()
  {
    MatrixOpsConfig config = {
      nullptr, // stream
      false,   // is_input_on_device
      false,   // is_output_on_device
      false,   // is_async
    };
    return config;
  }

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
  matrix_transpose(const E* mat_in, uint32_t nof_rows, uint32_t nof_cols, const MatrixOpsConfig& config, E* mat_out);

  // field specific APIs. TODO Yuval move to api headers like icicle V2
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, matrix_transpose)(
    const scalar_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const MatrixOpsConfig& config, scalar_t* mat_out);

  /*************************** Backend registration ***************************/

  using scalarMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const MatrixOpsConfig& config,
    scalar_t* out)>;

  void register_matrix_transpose(const std::string& deviceType, scalarMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool _reg_vec_add = []() -> bool {                                                                          \
      register_matrix_transpose(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle