
#include "icicle/matrix_ops.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

/*********************************** TRANSPOSE ***********************************/

template <typename T>
eIcicleError cpu_matrix_transpose(
  const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  // Check for invalid arguments
  if (!mat_in || !mat_out || nof_rows == 0 || nof_cols == 0) { return eIcicleError::INVALID_ARGUMENT; }

  // Perform the matrix transpose
  for (uint32_t i = 0; i < nof_rows; ++i) {
    for (uint32_t j = 0; j < nof_cols; ++j) {
      mat_out[j * nof_rows + i] = mat_in[i * nof_cols + j];
    }
  }

  return eIcicleError::SUCCESS;
}

REGISTER_MATRIX_TRANSPOSE_BACKEND("CPU", cpu_matrix_transpose<scalar_t>);
#ifdef EXT_FIELD
REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND("CPU", cpu_matrix_transpose<extension_t>);
#endif // EXT_FIEL
