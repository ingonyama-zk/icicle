#pragma once

#include "icicle/gate_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

  using GateEvaluationImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* constants,
    const scalar_t* fixed, 
    const scalar_t* advice,
    const scalar_t* instance,
    const scalar_t* beta,
    const scalar_t* gamma,
    const scalar_t* theta,
    const scalar_t* y,
    size_t num_elements,
    int rot_scale,
    int isize,
    const GateOpsConfig* config,
    scalar_t* results)>;


} // namespace icicle