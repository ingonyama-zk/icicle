#include "icicle/backend/gate_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** EVALUATE ************************/
  ICICLE_DISPATCHER_INST(GateEvaluationDispatcher, gate_evaluation, gateEvaluationImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, gate_evaluation)(
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
    scalar_t* results)
  {
    return GateEvaluationDispatcher::execute(
      constants,
      fixed,
      advice,
      instance,
      beta,
      gamma,
      theta,
      y,
      num_elements,
      rot_scale,
      isize,
      config,
      results
    );
  }

  template <>
  eIcicleError gate_evaluation(
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
    const GateOpsConfig& config,
    scalar_t* results)
  {
      return CONCAT_EXPAND(FIELD, gate_evaluation)(
          constants,
          fixed, 
          advice, 
          instance,
          beta, 
          gamma, 
          theta, 
          y,
          num_elements, 
          rot_scale, 
          isize,
          &config, 
          results
      );
  }


} // namespace icicle