#include "icicle/backend/vec_ops_backend.h"

template <typename T>   
static eIcicleError cpu_matrix_mul(
  const Device& device,
  const T* mat_a, uint32_t nof_rows_a, uint32_t nof_cols_a,
  const T* mat_b, uint32_t nof_rows_b, uint32_t nof_cols_b, const VecOpsConfig& config, T* mat_out)
{
  return eIcicleError::SUCCESS;
}   

REGISTER_MATRIX_MUL_BACKEND("CPU", cpu_matrix_mul<scalar_t>);