#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"
#include <cstdint>
#include <sys/types.h>
#include <vector>

#include "taskflow/taskflow.hpp"
#include "icicle/program/program.h"
#include "cpu_program_executor.h"
#include "util.h"

using namespace field_config;
using namespace icicle;

/*********************************** MATRIX MULTIPLICATION ***********************************/

template <typename T>
static eIcicleError cpu_matrix_mult(
  const Device& device,
  const T* mat_a,
  uint32_t nof_rows_a,
  uint32_t nof_cols_a,
  const T* mat_b,
  uint32_t nof_rows_b,
  uint32_t nof_cols_b,
  const VecOpsConfig& config,
  T* mat_out)
{
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_MATRIX_MULT_BACKEND("CPU", cpu_matrix_mult<scalar_t>);
#ifdef RING
    REGISTER_POLY_RING_MATRIX_MULT_BACKEND("CPU", cpu_matrix_mult<PolyRing>);

#endif
