
#include "icicle/vec_ops.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

/*********************************** ADD ***********************************/
template <typename T>
eIcicleError
cpu_vector_add(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = vec_a[i] + vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_ADD_BACKEND("CPU", cpu_vector_add<scalar_t>);

/*********************************** SUB ***********************************/
template <typename T>
eIcicleError
cpu_vector_sub(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = vec_a[i] - vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_SUB_BACKEND("CPU", cpu_vector_sub<scalar_t>);

/*********************************** MUL ***********************************/
template <typename T>
eIcicleError
cpu_vector_mul(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = vec_a[i] * vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_MUL_BACKEND("CPU", cpu_vector_mul<scalar_t>);

/*********************************** DIV ***********************************/
template <typename T>
eIcicleError
cpu_vector_div(const Device& device, const T* vec_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = vec_a[i] * T::inverse(vec_b[i]);
  }
  return eIcicleError::SUCCESS;
}

REGISTER_VECTOR_DIV_BACKEND("CPU", cpu_vector_div<scalar_t>);

/*********************************** MUL BY SCALAR***********************************/
template <typename T>
eIcicleError cpu_scalar_mul(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = *scalar_a * vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_SCALAR_MUL_BACKEND("CPU", cpu_scalar_mul<scalar_t>);

/*********************************** CONVERT MONTGOMERY ***********************************/
template <typename T>
eIcicleError cpu_convert_montgomery(
  const Device& device, const T* input, uint64_t n, bool is_into, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = is_into ? T::to_montgomery(input[i]) : T::from_montgomery(input[i]);
  }
  return eIcicleError::SUCCESS;
}

REGISTER_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_montgomery<scalar_t>);

#ifdef EXT_FIELD
REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND("CPU", cpu_vector_add<extension_t>);
REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND("CPU", cpu_vector_sub<extension_t>);
REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND("CPU", cpu_vector_mul<extension_t>);
REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND("CPU", cpu_convert_montgomery<extension_t>);
#endif // EXT_FIELD

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
#endif // EXT_FIELD

/*********************************** BIT REVERSE ***********************************/

template <typename T>
eIcicleError
cpu_bit_reverse(const Device& device, const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out)
{
  // Check for invalid arguments
  if (!vec_in || !vec_out || size == 0) { return eIcicleError::INVALID_ARGUMENT; }

  // Calculate log2(size)
  int logn = static_cast<int>(std::floor(std::log2(size)));
  if ((1ULL << logn) != size) {
    return eIcicleError::INVALID_ARGUMENT; // Ensure size is a power of 2
  }

  // If vec_in and vec_out are not the same, copy input to output
  if (vec_in != vec_out) {
    for (uint64_t i = 0; i < size; ++i) {
      vec_out[i] = vec_in[i];
    }
  }

  // Perform the bit reverse
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t rev = 0;
    for (int j = 0; j < logn; ++j) {
      if (i & (1ULL << j)) { rev |= 1ULL << (logn - 1 - j); }
    }
    if (i < rev) { std::swap(vec_out[i], vec_out[rev]); }
  }

  return eIcicleError::SUCCESS;
}

REGISTER_BIT_REVERSE_BACKEND("CPU", cpu_bit_reverse<scalar_t>);
#ifdef EXT_FIELD
REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND("CPU", cpu_bit_reverse<extension_t>);
#endif // EXT_FIELD

/*********************************** SLICE ***********************************/

template <typename T>
eIcicleError cpu_slice(
  const Device& device,
  const T* vec_in,
  uint64_t offset,
  uint64_t stride,
  uint64_t size,
  const VecOpsConfig& config,
  T* vec_out)
{
  if (vec_in == nullptr || vec_out == nullptr) {
    ICICLE_LOG_ERROR << "Error: Invalid argument - input or output vector is null";
    return eIcicleError::INVALID_ARGUMENT;
  }

  for (uint64_t i = 0; i < size; ++i) {
    uint64_t index = offset + i * stride;
    vec_out[i] = vec_in[index];
  }

  return eIcicleError::SUCCESS;
}

REGISTER_SLICE_BACKEND("CPU", cpu_slice<scalar_t>);
#ifdef EXT_FIELD
REGISTER_SLICE_EXT_FIELD_BACKEND("CPU", cpu_slice<extension_t>);
#endif // EXT_FIELD