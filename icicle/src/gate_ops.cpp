#include "icicle/backend/gate_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** EVALUATE ************************/
  ICICLE_DISPATCHER_INST(GateEvaluationDispatcher, gate_evaluation, gateEvaluationImpl);
  ICICLE_DISPATCHER_INST(LookupsConstraintDispatcher, lookups_constraint, lookupsConstraintImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, gate_evaluation)(
    const GateData<scalar_t>* gate_data, 
    const CalculationData<scalar_t>* calc_data,
    const HornerData* horner_data,
    const GateOpsConfig* config,
    scalar_t* results)
  {
      return GateEvaluationDispatcher::execute(
        *gate_data, 
        *calc_data,
        *horner_data,
        *config,
        results
      );
  }

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, lookups_constraint)(
    const LookupData<scalar_t>* lookup_data, 
    const LookupConfig* config,
    scalar_t* results)
  {
      return LookupsConstraintDispatcher::execute(
        *lookup_data, 
        *config,
        results
      );
  }

  eIcicleError gate_evaluation(
    const GateData<scalar_t>& gate_data, 
    const CalculationData<scalar_t>& calc_data,
    const HornerData& horner_data,
    const GateOpsConfig& config,
    scalar_t* results)
  {
    return CONCAT_EXPAND(FIELD, gate_evaluation)(
      &gate_data, 
      &calc_data,
      &horner_data,
      &config,
      results
    );
  }

  eIcicleError lookups_constraint(
    const LookupData<scalar_t>& lookup_data, 
    const LookupConfig& config,
    scalar_t* results)
  {
    return CONCAT_EXPAND(FIELD, lookups_constraint)(
      &lookup_data, 
      &config,
      results
    );
  }

}
