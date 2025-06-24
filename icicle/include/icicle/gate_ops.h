#pragma once

#include <cstdint>
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
    bool is_calculations_on_device; /** True if `calculations` array is on the device. Default: false. */
    bool is_previous_value_on_device;   /** True if `previous_value` array is on the device. Default: false. */
    bool is_horners_on_device;      /** True if `horners` array is on the device. Default: false. */
    bool is_result_on_device;       /** True to keep results on device. Default: false. */
    bool is_async;                  /** Async execution flag. Default: false. */
  };


  struct LookupConfig {
      icicleStreamHandle stream;          /** Stream for asynchronous execution. */
      bool is_table_values_on_device;     /** True if `table_values` columns are on the device. Default: false. */
      bool is_inputs_prods_on_device;     /** True if `inputs_prods` is on the device. Default: false. */
      bool is_inputs_inv_sums_on_device;  /** True if `inputs_inv_sums` is on the device. Default: false. */
      bool is_coset_on_device;     /** True if `phi_coset` and `m_coset` columns are on the device. Default: false. */
      bool is_l_on_device;                /** True if `l0`, `l_last` and `l_active_row` columns are on the device. Default: false. */
      bool is_y_on_device;                /** True if `y` array is on the device. Default: false. */
      bool is_previous_value_on_device;   /** True if `previous_value` array is on the device. Default: false. */
      bool is_result_on_device;           /** True to keep results on device. Default: false. */
      bool is_async;
  };

  struct HornerData {
    const uint32_t* value_types;      // Value types for Horner's method
    const uint32_t* value_indices;  // Source indices for Horner's method
    const uint32_t* offsets;          // Horner offsets
    const uint32_t* sizes;            // Horner sizes
    uint32_t num_horner;
  };

  template <typename T>
  struct CalculationData {
    const uint32_t* calc_types;     // Array for calculation types (e.g., Add, Sub, Mul, etc.)
    const uint32_t* targets;         // Source value types (e.g., beta, advice, etc.)
    const uint32_t* value_types;    // Source value types (e.g., beta, advice, etc.)
    const uint32_t* value_indices;  // Source value indices
    const T* constants;          // Constants array
    uint32_t num_constants;        // Number of constants
    const int32_t* rotations;        // Rotations array
    uint32_t num_rotations;        // Number of rotations
    const T* previous_value;
    bool is_prev_zero;
    uint32_t num_calculations;      // Number of calculations
    uint32_t num_intermediates;     // Number of intermediate values
    uint32_t num_elements;
    uint32_t rot_scale;               // Rotation scale
    uint32_t i_size;                  // Size of i
  };

  template <typename T>
  struct GateData {
    const T* fixed;              // Fixed columns array
    uint32_t num_fixed_columns;    // Number of fixed columns
    uint32_t num_fixed_rows;    // Number of fixed columns
    const T* advice;             // Advice columns array
    uint32_t num_advice_columns;   // Number of advice columns
    uint32_t num_advice_rows;   // Number of advice columns
    const T* instance;           // Instance columns array
    uint32_t num_instance_columns; // Number of instance columns
    uint32_t num_instance_rows; // Number of instance columns
    const T* challenges;         // Challenges array
    uint32_t num_challenges;       // Number of challenges
    const T* beta;
    const T* gamma;
    const T* theta;
    const T* y;
  };

  template <typename T>
  struct LookupData {
    const T* table_values;
    uint32_t num_table_values;
    const T* inputs_prods;
    uint32_t num_inputs_prods;
    const T* inputs_inv_sums;
    uint32_t num_inputs_inv_sums;
    const T* phi_coset;
    uint32_t num_phi_coset;
    const T* m_coset;
    uint32_t num_m_coset;
    const T* l0;
    uint32_t num_l0;
    const T* l_last;
    uint32_t num_l_last;
    const T* l_active_row;
    uint32_t num_l_active_row;
    const T* y;
    const T* previous_value;
    uint32_t num_elements;
    uint32_t rot_scale;
    uint32_t i_size;
  };

  /**
   * @brief Returns the default value of GateOpsConfig.
   */
  static GateOpsConfig default_gate_ops_config()
  {
    return GateOpsConfig{
      nullptr,  // stream
      false,    // is_constants_on_device
      false,    // is_fixed_on_device
      false,    // is_advice_on_device
      false,    // is_instance_on_device
      false,    // is_rotations_on_device
      false,    // is_challenges_on_device
      false,    // is_calculations_on_device
      false,    // is_horners_on_device
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
    const CalculationData<T>& calc_data,
    const HornerData& horner_data,
    const GateOpsConfig& config, 
    T* results
  );

  /**
   * @brief Evaluate the lookups constraints with full parameter set
   */
  template <typename T>
  eIcicleError lookups_constraint(
    const LookupData<T>& gate_data, 
    const LookupConfig& config, 
    T* results
  );

} // namespace icicle