#include "icicle/jl_projection.h"
#include "icicle/backend/vec_ops_backend.h" // defines type field_t
#include "icicle/dispatcher.h"

namespace icicle {

  // === Dispatcher Instantiations ===

  ICICLE_DISPATCHER_INST(JLProjectionDispatcher, jl_projection, JLProjectionImpl);
  ICICLE_DISPATCHER_INST(JLProjectionGetRowsDispatcher, jl_projection_get_rows, JLProjectionGetRowsImpl);

  // === Scalar (Zq) Projection API ===

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

  // === Scalar (Zq) JL Row Generation ===

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection_get_rows)(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    const VecOpsConfig* cfg,
    field_t* output)
  {
    return JLProjectionGetRowsDispatcher::execute(
      seed, seed_len,
      row_size, // Zq entries per row
      start_row, num_rows,
      false, // conjugate = false
      0,     // conjugate_poly_size_d = 0
      *cfg, output);
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

  // === Polynomial (Rq) JL Row Generation ===

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection_get_rows_polyring)(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size, // number of Rq polynomials per row
    size_t start_row,
    size_t num_rows,
    bool conjugate,
    const VecOpsConfig* cfg,
    PolyRing* output)
  {
    // Map Rq rows to scalar buffer (row-major): num_rows × row_size × d
    field_t* scalar_output = reinterpret_cast<field_t*>(output);
    const size_t scalar_row_size = PolyRing::d * row_size;

    return JLProjectionGetRowsDispatcher::execute(
      seed, seed_len,
      scalar_row_size, // number of Zq scalars per row
      start_row, num_rows, conjugate,
      PolyRing::d, // conjugate_poly_size_d = PolyRing degree
      *cfg, scalar_output);
  }

  template <>
  eIcicleError get_jl_matrix_rows(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    bool conjugate,
    const VecOpsConfig& cfg,
    PolyRing* output)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, jl_projection_get_rows_polyring)(
      seed, seed_len, row_size, start_row, num_rows, conjugate, &cfg, output);
  }

} // namespace icicle