
#include "icicle/balanced_decomposition.h"
#include "icicle/backend/vec_ops_backend.h" // defines type field_t
#include "icicle/dispatcher.h"

namespace icicle {

  static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");

  /*********************************** BALANCED DECOMPOSITION/RECOMPOSITION ************************/
  ICICLE_DISPATCHER_INST(BalancedDecomposeDispatcher, decompose_balanced_digits, balancedDecompositionImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, decompose_balanced_digits)(
    const field_t* input,
    size_t input_size,
    uint32_t base,
    const VecOpsConfig* config,
    field_t* output,
    size_t output_size)
  {
    return BalancedDecomposeDispatcher::execute(input, input_size, base, *config, output, output_size);
  }

  namespace balanced_decomposition {
    template <>
    eIcicleError decompose(
      const field_t* input,
      size_t input_size,
      uint32_t base,
      const VecOpsConfig& config,
      field_t* output,
      size_t output_size)
    {
      return CONCAT_EXPAND(ICICLE_FFI_PREFIX, decompose_balanced_digits)(
        input, input_size, base, &config, output, output_size);
    }
  } // namespace balanced_decomposition

  ICICLE_DISPATCHER_INST(BalancedRecomposeDispatcher, recompose_from_balanced_digits, balancedDecompositionImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, recompose_from_balanced_digits)(
    const field_t* input,
    size_t input_size,
    uint32_t base,
    const VecOpsConfig* config,
    field_t* output,
    size_t output_size)
  {
    return BalancedRecomposeDispatcher::execute(input, input_size, base, *config, output, output_size);
  }

  namespace balanced_decomposition {
    template <>
    eIcicleError recompose(
      const field_t* input,
      size_t input_size,
      uint32_t base,
      const VecOpsConfig& config,
      field_t* output,
      size_t output_size)
    {
      return CONCAT_EXPAND(ICICLE_FFI_PREFIX, recompose_from_balanced_digits)(
        input, input_size, base, &config, output, output_size);
    }
  } // namespace balanced_decomposition

  extern "C" uint32_t CONCAT_EXPAND(ICICLE_FFI_PREFIX, balanced_decomposition_nof_digits)(uint32_t base)
  {
    return balanced_decomposition::compute_nof_digits<field_t>(base);
  }

} // namespace icicle