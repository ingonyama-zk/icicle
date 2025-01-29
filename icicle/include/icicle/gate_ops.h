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
    bool is_calculations_on_device;
    bool is_result_on_device;       /** True to keep results on device. Default: false. */
    bool is_async;                  /** Async execution flag. Default: false. */
    ConfigExtension* ext = nullptr; /** Backend-specific extension. */
  };

  struct HornerData {
    const int* value_types;      // Value types for Horner's method
    const int* i_value_indices;  // Source indices for Horner's method
    const int* j_value_indices;  // Secondary indices for Horner's method
    const int* offsets;          // Horner offsets
    const int* sizes;            // Horner sizes
  };

  struct CalculationData {
    const int* value_types;     // Array for calculation types (e.g., Add, Sub, Mul, etc.)
    const int* i_value_types;    // Source value types (e.g., beta, advice, etc.)
    const int* j_value_types;    // Secondary source value types
    const int* i_value_indices;  // Source value indices
    const int* j_value_indices;  // Secondary source value indices
    size_t num_calculations;     // Number of calculations
    size_t num_intermediates;    // Number of intermediate values
  };

  template <typename T>
  struct GateData {
    const T* constants;          // Constants array
    size_t num_constants;        // Number of constants
    const T* fixed;              // Fixed columns array
    size_t num_fixed_columns;    // Number of fixed columns
    const T* advice;             // Advice columns array
    size_t num_advice_columns;   // Number of advice columns
    const T* instance;           // Instance columns array
    size_t num_instance_columns; // Number of instance columns
    const int* rotations;        // Rotations array
    size_t num_rotations;        // Number of rotations
    const T* challenges;         // Challenges array
    size_t num_challenges;       // Number of challenges
    const T* beta;
    const T* gamma;
    const T* theta;
    const T* y;
    const T* previous_value;
    int num_elements;
    int rot_scale;               // Rotation scale
    int i_size;                  // Size of i
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
      false,
      false,    // is_result_on_device
      false     // is_async
    };
  }

  /**
   * @brief Evaluate the gates with full parameter set
   */
  template <typename T>
  eIcicleError gate_evaluation(
    const GateData<T>& gate_data, 
    const CalculationData& calc_data,
    const HornerData& horner_data,
    const GateOpsConfig& config, 
    T* results
  );

} // namespace icicle