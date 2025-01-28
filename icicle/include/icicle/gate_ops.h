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
    bool is_constants_on_device;    /** True if `constants` is on the device. Default: false. */
    bool is_fixed_on_device;        /** True if `fixed` columns are on the device. Default: false. */
    bool is_advice_on_device;       /** True if `advice` columns are on the device. Default: false. */
    bool is_instance_on_device;     /** True if `instance` columns are on the device. Default: false. */
    bool is_rotations_on_device;    /** True if `rotations` array is on the device. Default: false. */
    bool is_challenges_on_device;   /** True if `challenges` array is on the device. Default: false. */
    bool is_result_on_device;       /** True to keep results on device. Default: false. */
    bool is_async;                  /** Async execution flag. Default: false. */
    ConfigExtension* ext = nullptr; /** Backend-specific extension. */
  };

  /**
   * @brief Returns the default value of GateOpsConfig.
   */
  static GateOpsConfig default_vec_ops_config()
  {
    return GateOpsConfig{
      nullptr,  // stream
      false,    // is_constants_on_device
      false,    // is_fixed_on_device
      false,    // is_advice_on_device
      false,    // is_instance_on_device
      false,    // is_rotations_on_device
      false,    // is_challenges_on_device
      false,    // is_result_on_device
      false     // is_async
    };
  }

  /**
   * @brief Evaluate the gates with full parameter set
   */
  template <typename T>
  eIcicleError gate_evaluation(
    const T* constants,
    size_t num_constants,
    const T* fixed,
    size_t num_fixed_columns,
    const T* advice,
    size_t num_advice_columns,
    const T* instance,
    size_t num_instance_columns,
    const T* challenges,
    size_t num_challenges,
    const int* rotations,
    size_t num_rotations,
    const T* beta,
    const T* gamma,
    const T* theta,
    const T* y,
    const T* previous_value,
    const int* calculations,            // Array for calculation types (e.g., Add, Sub, Mul, etc.)
    const int* i_value_types,           // Source value types (e.g., beta, advice, etc.)
    const int* j_value_types,           // Secondary source value types
    const int* i_value_indices,         // Source value indices
    const int* j_value_indices,         // Secondary source value indices
    const int* horner_value_types,      // Value types for Horner's method
    const int* i_horner_value_indices,  // Source indices for Horner's method
    const int* j_horner_value_indices,  // Secondary indices for Horner's method
    const int* horner_offsets,          // Horner offsets
    const int* horner_sizes,            // Horner sizes
    size_t num_calculations,            // Number of calculations
    size_t num_intermediates,           // Number of intermediate values
    size_t num_elements,
    int rot_scale,
    int isize,
    const GateOpsConfig& config, 
    T* results
  );

} // namespace icicle