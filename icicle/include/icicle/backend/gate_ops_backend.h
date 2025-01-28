#pragma once

#include "icicle/gate_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

  using gateEvaluationImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* constants,
    size_t num_constants,
    const scalar_t* fixed,
    size_t num_fixed_columns,
    const scalar_t* advice,
    size_t num_advice_columns,
    const scalar_t* instance,
    size_t num_instance_columns,
    const scalar_t* challenges,
    size_t num_challenges,
    const int* rotations,
    size_t num_rotations,
    const scalar_t* beta,
    const scalar_t* gamma,
    const scalar_t* theta,
    const scalar_t* y,
    const scalar_t* previous_value,
    const int* calculations,
    const int* i_value_types,
    const int* j_value_types,
    const int* i_value_indices,
    const int* j_value_indices,
    const int* horner_value_types,
    const int* i_horner_value_indices,
    const int* j_horner_value_indices,
    const int* horner_offsets,
    const int* horner_sizes,
    size_t num_calculations,
    size_t num_intermediates,
    size_t num_elements,
    int rot_scale,
    int isize,
    const GateOpsConfig* config,
    scalar_t* results)>;


  void register_gate_evaluation(const std::string& deviceType, gateEvaluationImpl impl);

#define REGISTER_GATE_EVALUATION_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_gate_evaluation) = []() -> bool {                                                      \
      register_gate_evaluation(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }


} // namespace icicle