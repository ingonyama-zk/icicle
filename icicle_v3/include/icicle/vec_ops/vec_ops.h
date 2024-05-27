#pragma once

#include <functional>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include "errors.h"
#include "runtime.h"

#include "fields/field.h"
#include "fields/field_config.h"
#include "utils/utils.h"

using namespace field_config;

namespace icicle {

  /*************************** CONFIG ***************************/
  struct VecOpsConfig {
    bool is_a_on_device;      /**< True if `a` is on device and false if it is not. Default value: false. */
    bool is_b_on_device;      /**< True if `b` is on device and false if it is not. Default value: false. */
    bool is_result_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */
    icicleStreamHandle stream; /**< stream for async execution. */
  };

  /**
   * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
   * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
   */
  static VecOpsConfig DefaultVecOpsConfig()
  {
    VecOpsConfig config = {
      false,   // is_a_on_device
      false,   // is_b_on_device
      false,   // is_result_on_device
      false,   // is_async
      nullptr, // stream
    };
    return config;
  }

  /*************************** APIs ***************************/
  // Template alias for a function implementing vector addition for a specific device and type T
  template <typename T>
  using VectorAddImpl = std::function<eIcicleError(
    const Device& device, const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)>;

  // Declaration of the vector addition function for integer vectors
  // This function performs element-wise addition of two integer vectors on a specified device
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, VectorAdd)(
    const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output);

  // This function allows C++ code to call to VectorAdd agnostic of the field
  static inline eIcicleError
  VectorAdd(const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, VectorAdd)(vec_a, vec_b, n, config, output);
  }

  /*************************** REGISTRATION ***************************/
  // Function to register a vector addition implementation for a specific device type
  // This allows the system to use the appropriate implementation based on the device type
  extern "C" void registerVectorAdd(const std::string& deviceType, VectorAddImpl<scalar_t> impl);

// Macro to simplify the registration of a vector addition implementation for a device type
// Usage: REGISTER_VECTOR_ADD_BACKEND("device_type", implementation_function)
#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool _reg_vec_add = []() -> bool {                                                                          \
      registerVectorAdd(DEVICE_TYPE, FUNC);                                                                            \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle