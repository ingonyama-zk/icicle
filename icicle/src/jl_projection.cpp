#include "icicle/jl_projection.h"
#include "icicle/backend/vec_ops_backend.h" // defines type field_t
#include "icicle/dispatcher.h"

namespace icicle {
  ICICLE_DISPATCHER_INST(JLProjectionDispatcher, jl_projection, JLProjectionImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection)(
    const field_t* input,
    size_t N,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig* cfg,
    field_t* output // length 256
  )
  {
    return JLProjectionDispatcher::execute(input, N, seed, seed_len, *cfg, output);
  }
} // namespace icicle