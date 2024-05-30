
#include "icicle/vec_ops.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field_config.h"

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

/*********************************** CONVERT MONTGOMERY ***********************************/
template <typename T>
eIcicleError cpu_convert_montgomery(const Device& device, T* vec, uint64_t n, bool is_into, const VecOpsConfig& config)
{
  std::cerr << "cpu_convert_montgomery() not implemented" << std::endl;
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_montgomery<scalar_t>);
