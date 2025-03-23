
#include "icicle/balanced_decomposition.h"
#include "icicle/backend/vec_ops_backend.h" // defines type field_t
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** BALANCED DECOMPOSITION/RECOMPOSITION ************************/
  ICICLE_DISPATCHER_INST(BalancedDecomposeDispatcher, decompose_balanced_digits, balancedDecompositionImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, decompose_balanced_digits)(
    const field_t* input, size_t input_size, uint32_t base, const VecOpsConfig* config, field_t* output)
  {
    return BalancedDecomposeDispatcher::execute(input, input_size, base, *config, output);
  }

  template <>
  eIcicleError decompose_balanced_digits(
    const field_t* input, size_t input_size, uint32_t base, const VecOpsConfig& config, field_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, decompose_balanced_digits)(input, input_size, base, &config, output);
  }

  ICICLE_DISPATCHER_INST(BalancedRecomposeDispatcher, recompose_from_balanced_digits, balancedDecompositionImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, recompose_from_balanced_digits)(
    const field_t* input, size_t output_size, uint32_t base, const VecOpsConfig* config, field_t* output)
  {
    return BalancedRecomposeDispatcher::execute(input, output_size, base, *config, output);
  }

  template <>
  eIcicleError recompose_from_balanced_digits(
    const field_t* input, size_t output_size, uint32_t base, const VecOpsConfig& config, field_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, recompose_from_balanced_digits)(input, output_size, base, &config, output);
  }

} // namespace icicle