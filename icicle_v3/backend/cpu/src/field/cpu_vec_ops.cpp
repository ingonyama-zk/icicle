
#include "icicle/vec_ops/vec_ops.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "fields/field_config.h"

using namespace field_config;
using namespace icicle;

/*********************************** ADD ***********************************/
template <typename T>
eIcicleError
cpu_vector_add(const Device& device, const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
{
  for (int i = 0; i < n; ++i) {
    output[i] = vec_a[i] + vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_ADD_BACKEND("CPU", cpu_vector_add<scalar_t>);

/*********************************** SUB ***********************************/
template <typename T>
eIcicleError
cpu_vector_sub(const Device& device, const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
{
  for (int i = 0; i < n; ++i) {
    output[i] = vec_a[i] - vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_SUB_BACKEND("CPU", cpu_vector_sub<scalar_t>);

/*********************************** MUL ***********************************/
template <typename T>
eIcicleError
cpu_vector_mul(const Device& device, const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
{
  for (int i = 0; i < n; ++i) {
    output[i] = vec_a[i] * vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_MUL_BACKEND("CPU", cpu_vector_mul<scalar_t>);