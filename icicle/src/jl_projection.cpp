#include "icicle/jl_projection.h"
#include "icicle/backend/vec_ops_backend.h" // defines type field_t
#include "icicle/dispatcher.h"

namespace icicle {
  ICICLE_DISPATCHER_INST(JLProjectionDispatcher, jl_projection, JLProjectionImpl);
  ICICLE_DISPATCHER_INST(JLProjectionGetRowsDispatcher, jl_projection_get_rows, JLProjectionGetRowsImpl);

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

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection_get_rows)(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    const VecOpsConfig* cfg,
    field_t* output)
  {
    return JLProjectionGetRowsDispatcher::execute(seed, seed_len, row_size, start_row, num_rows, *cfg, output);
  }

  template <>
  eIcicleError get_jl_matrix_rows(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    const VecOpsConfig& cfg,
    field_t* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection_get_rows)(
      seed, seed_len, row_size, start_row, num_rows, &cfg, output);
  }
} // namespace icicle