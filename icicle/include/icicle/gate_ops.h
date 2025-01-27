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
   * @brief Configuration for gate operations.
   */
  struct GateOpsConfig {
    icicleStreamHandle stream;      /** Stream for asynchronous execution. */
    bool is_constants_on_device;    /** True if `constants` is on the device, false if it is not. Default value: false. */
    bool is_fixed_on_device;        /** True if `fixed` is on the device, false if it is not. Default value: false. */
    bool is_advice_on_device;       /** True if `advice` is on the device, false if it is not. Default value: false. */
    bool is_instance_on_device;     /** True if `instance` is on the device, false if it is not. Default value: false. */
    bool is_result_on_device;       /** If true, the output is preserved on the device, otherwise on the host. Default value:
                                        false. */
    bool is_async;                  /** Whether to run the vector operations asynchronously.
                                        If set to `true`, the function will be non-blocking and synchronization
                                        must be explicitly managed using `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                        If set to `false`, the function will block the current CPU thread. */
    ConfigExtension* ext = nullptr; /** Backend-specific extension. */
  };

  /**
   * @brief Returns the default value of GateOpsConfig.
   *
   * @return Default value of GateOpsConfig.
   */
  static GateOpsConfig default_vec_ops_config()
  {
    GateOpsConfig config = {
      nullptr, // stream
      false,   // is_constants_on_device
      false,   // is_fixed_on_device
      false,   // is_advice_on_device
      false,   // is_instance_on_device
      false,   // is_result_on_device
      false,   // is_async
    };
    return config;
  }

  /**
   * @brief Evaluate the gates.
   *
   */
  template <typename T>
  eIcicleError
  gate_evaluation(
    const T* constants, 
    const T* fixed,
    const T* advice,
    const T* instance,
    const T* beta,
    const T* gamma,
    const T* theta,
    const T* y,
    size_t num_elements,
    int rot_scale,
    int isize,
    const GateOpsConfig& config, 
    T* results
  );

} // namespace icicle