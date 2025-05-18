#include "icicle/jl_projection.h"
#include "icicle/backend/vec_ops_backend.h" // defines type field_t
#include "icicle/dispatcher.h"

namespace icicle {
  ICICLE_DISPATCHER_INST(JLProjectionDispatcher, jl_projection, JLProjectionImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection)(
    const field_t* input,
    size_t input_size,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig* cfg,
    field_t* output,
    size_t output_size)
  {
    return JLProjectionDispatcher::execute(input, input_size, seed, seed_len, *cfg, output, output_size);
  }

  template <>
  eIcicleError jl_projection(
    const field_t* input,
    size_t input_size,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& cfg,
    field_t* output,
    size_t output_size)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection)(
      input, input_size, seed, seed_len, &cfg, output, output_size);
  }
} // namespace icicle