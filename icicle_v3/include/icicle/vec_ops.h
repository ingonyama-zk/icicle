#pragma once

#include <functional>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include "errors.h"
#include "runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

using namespace field_config;

namespace icicle {

  /*************************** CONFIG ***************************/
  struct VecOpsConfig {
    icicleStreamHandle stream; /**< stream for async execution. */
    bool is_a_on_device;       /**< True if `a` is on device and false if it is not. Default value: false. */
    bool is_b_on_device;       /**< True if `b` is on device and false if it is not. Default value: false. */
    bool is_result_on_device;  /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */

    ConfigExtension ext; /** backend specific extensions*/
  };

  /**
   * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
   * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
   */
  static VecOpsConfig default_vec_ops_config()
  {
    VecOpsConfig config = {
      nullptr, // stream
      false,   // is_a_on_device
      false,   // is_b_on_device
      false,   // is_result_on_device
      false,   // is_async
    };
    return config;
  }

  /*************************** API TYPE ***************************/
  using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* vec_a,
    const scalar_t* vec_b,
    int n,
    const VecOpsConfig& config,
    scalar_t* output)>;

  /*************************** ADD ***************************/
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_add)(
    const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output);

  static inline eIcicleError
  vector_add(const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_add)(vec_a, vec_b, n, config, output);
  }

  extern "C" void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool _reg_vec_add = []() -> bool {                                                                          \
      register_vector_add(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  /*************************** SUB ***************************/
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_sub)(
    const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output);

  static inline eIcicleError
  vector_sub(const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_sub)(vec_a, vec_b, n, config, output);
  }

  extern "C" void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool _reg_vec_sub = []() -> bool {                                                                          \
      register_vector_sub(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  /*************************** MUL ***************************/
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_mul)(
    const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output);

  static inline eIcicleError
  vector_mul(const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, vector_mul)(vec_a, vec_b, n, config, output);
  }

  extern "C" void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool _reg_vec_mul = []() -> bool {                                                                          \
      register_vector_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle