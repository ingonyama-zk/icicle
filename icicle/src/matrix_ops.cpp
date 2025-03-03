#include "icicle/backend/vec_ops_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {
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