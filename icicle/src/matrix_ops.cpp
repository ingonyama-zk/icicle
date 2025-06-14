#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {
  /*********************************** MATRIX  MULTIPLICATION *************************/

  ICICLE_DISPATCHER_INST(MatrixMultiplicationDispatcher, matrix_mult, scalarBinaryMatrixOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, matrix_mult)(
    const scalar_t* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig* config,
    scalar_t* mat_out)
  {
    return MatrixMultiplicationDispatcher::execute(
      mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, *config, mat_out);
  }

  template <>
  eIcicleError matrix_mult(
    const scalar_t* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    scalar_t* mat_out)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, matrix_mult)(
      mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, &config, mat_out);
  }

#ifdef RING
  ICICLE_DISPATCHER_INST(PolyRingMatrixMultiplicationDispatcher, poly_ring_matrix_mult, polyRingBinaryMatrixOpImpl);
  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, poly_ring_matrix_mult)(
    uint32_t degree,
    const scalar_t* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig* config,
    scalar_t* mat_out)
  {
    return PolyRingMatrixMultiplicationDispatcher::execute(
      degree, mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, *config, mat_out);
  }

  template <>
  eIcicleError poly_ring_matrix_mult(
    uint32_t degree,
    const scalar_t* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    scalar_t* mat_out)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, poly_ring_matrix_mult)(
      degree, mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, &config, mat_out);
  }
#endif

  /*********************************** TRANSPOSE ***********************************/
  ICICLE_DISPATCHER_INST(MatrixTransposeDispatcher, matrix_transpose, scalarMatrixOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, matrix_transpose)(
    const scalar_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig* config, scalar_t* mat_out)
  {
    return MatrixTransposeDispatcher::execute(mat_in, nof_rows, nof_cols, *config, mat_out);
  }

  template <>
  eIcicleError matrix_transpose(
    const scalar_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, scalar_t* mat_out)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, matrix_transpose)(mat_in, nof_rows, nof_cols, &config, mat_out);
  }

#ifdef EXT_FIELD
  ICICLE_DISPATCHER_INST(extFieldMatrixTransposeDispatcher, extension_matrix_transpose, extFieldMatrixOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_matrix_transpose)(
    const extension_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig* config, extension_t* mat_out)
  {
    return extFieldMatrixTransposeDispatcher::execute(mat_in, nof_rows, nof_cols, *config, mat_out);
  }

  template <>
  eIcicleError matrix_transpose(
    const extension_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, extension_t* mat_out)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_matrix_transpose)(mat_in, nof_rows, nof_cols, &config, mat_out);
  }
#endif // EXT_FIELD

#ifdef RING
  ICICLE_DISPATCHER_INST(ringRnsMatrixTransposeDispatcher, ring_rns_matrix_transpose, ringRnsMatrixOpImpl);

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_matrix_transpose)(
    const scalar_rns_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig* config, scalar_rns_t* mat_out)
  {
    return ringRnsMatrixTransposeDispatcher::execute(mat_in, nof_rows, nof_cols, *config, mat_out);
  }

  template <>
  eIcicleError matrix_transpose(
    const scalar_rns_t* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, scalar_rns_t* mat_out)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_matrix_transpose)(mat_in, nof_rows, nof_cols, &config, mat_out);
  }

#endif // RING
} // namespace icicle