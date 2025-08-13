#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

  /*********************************** NORM CHECKING ****************************************/
  ICICLE_DISPATCHER_INST(NormCheckDispatcher, check_norm_bound, normCheckImpl);
  ICICLE_DISPATCHER_INST(NormCheckRelativeDispatcher, check_norm_relative, normCheckRelativeImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, check_norm_bound)(
    const field_t* input, size_t size, eNormType norm, uint64_t norm_bound, const VecOpsConfig* config, bool* output)
  {
    return NormCheckDispatcher::execute(input, size, norm, norm_bound, *config, output);
  }

  namespace norm {
    template <>
    eIcicleError check_norm_bound(
      const field_t* input, size_t size, eNormType norm, uint64_t norm_bound, const VecOpsConfig& config, bool* output)
    {
      return CONCAT_EXPAND(ICICLE_FFI_PREFIX, check_norm_bound)(input, size, norm, norm_bound, &config, output);
    }
  } // namespace norm

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, check_norm_relative)(
    const field_t* input_a,
    const field_t* input_b,
    size_t size,
    eNormType norm,
    uint64_t scale,
    const VecOpsConfig* config,
    bool* output)
  {
    return NormCheckRelativeDispatcher::execute(input_a, input_b, size, norm, scale, *config, output);
  }

  namespace norm {
    template <>
    eIcicleError check_norm_relative(
      const field_t* input_a,
      const field_t* input_b,
      size_t size,
      eNormType norm,
      uint64_t scale,
      const VecOpsConfig& config,
      bool* output)
    {
      return CONCAT_EXPAND(ICICLE_FFI_PREFIX, check_norm_relative)(
        input_a, input_b, size, norm, scale, &config, output);
    }
  } // namespace norm

} // namespace icicle
