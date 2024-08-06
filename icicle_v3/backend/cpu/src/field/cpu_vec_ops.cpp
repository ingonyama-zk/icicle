
#include "icicle/backend/vec_ops_backend.h"
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

REGISTER_SCALAR_MUL_VEC_BACKEND("CPU", cpu_scalar_mul<scalar_t>);

/*********************************** Scalar + Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_add(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = *scalar_a + vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_SCALAR_ADD_VEC_BACKEND("CPU", cpu_scalar_add<scalar_t>);

/*********************************** Scalar - Vector***********************************/
template <typename T>
eIcicleError cpu_scalar_sub(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t n, const VecOpsConfig& config, T* output)
{
  for (uint64_t i = 0; i < n; ++i) {
    output[i] = *scalar_a - vec_b[i];
  }
  return eIcicleError::SUCCESS;
}

REGISTER_SCALAR_SUB_VEC_BACKEND("CPU", cpu_scalar_sub<scalar_t>);

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

/*********************************** Polynomial evaluation ***********************************/

template <typename T>
eIcicleError cpu_poly_eval(
  const Device& device,
  const T* coeffs,
  uint64_t coeffs_size,
  const T* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  T* evals /*OUT*/)
{
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
    evals[eval_idx] = coeffs[coeffs_size - 1];
    for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
      evals[eval_idx] = evals[eval_idx] * domain[eval_idx] + coeffs[coeff_idx];
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_EVAL("CPU", cpu_poly_eval<scalar_t>);

/*********************************** Highest non-zero idx ***********************************/
template <typename T>
eIcicleError cpu_highest_non_zero_idx(
  const Device& device, const T* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)
{
  *out_idx = -1; // zero vector is considered '-1' since 0 would be zero in vec[0]
  for (int64_t i = size - 1; i >= 0; --i) {
    if (input[i] != T::zero()) {
      *out_idx = i;
      break;
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND("CPU", cpu_highest_non_zero_idx<scalar_t>);

/*============================== polynomial division ==============================*/
template <typename T>
void school_book_division_step_cpu(T* r, T* q, const T* b, int deg_r, int deg_b, const T& lc_b_inv)
{
  int64_t monomial = deg_r - deg_b; // monomial=1 is 'x', monomial=2 is x^2 etc.

  T lc_r = r[deg_r];
  T monomial_coeff = lc_r * lc_b_inv; // lc_r / lc_b

  // adding monomial s to q (q=q+s)
  q[monomial] = monomial_coeff;

  for (int i = monomial; i <= deg_r; ++i) {
    T b_coeff = b[i - monomial];
    r[i] = r[i] - monomial_coeff * b_coeff;
  }
}

template <typename T>
eIcicleError cpu_poly_divide(
  const Device& device,
  const T* numerator,
  int64_t numerator_deg,
  const T* denumerator,
  int64_t denumerator_deg,
  const VecOpsConfig& config,
  T* q_out /*OUT*/,
  uint64_t q_size,
  T* r_out /*OUT*/,
  uint64_t r_size)
{
  ICICLE_ASSERT(r_size >= (1 + denumerator_deg))
    << "polynomial division expects r(x) size to be similar to numerator(x)";
  ICICLE_ASSERT(q_size >= (numerator_deg - denumerator_deg + 1))
    << "polynomial division expects q(x) size to be at least deg(numerator)-deg(denumerator)+1";

  ICICLE_CHECK(icicle_copy_async(r_out, numerator, (1 + numerator_deg) * sizeof(T), config.stream));

  // invert largest coeff of b
  const T& lc_b_inv = T::inverse(denumerator[denumerator_deg]);

  int64_t deg_r = numerator_deg;
  while (deg_r >= denumerator_deg) {
    // each iteration is removing the largest monomial in r until deg(r)<deg(b)
    school_book_division_step_cpu(r_out, q_out, denumerator, deg_r, denumerator_deg, lc_b_inv);

    // compute degree of r
    auto degree_config = default_vec_ops_config();
    cpu_highest_non_zero_idx(device, r_out, deg_r + 1 /*size of R*/, degree_config, &deg_r);
  }

  return eIcicleError::SUCCESS;
}

REGISTER_POLYNOMIAL_DIVISION("CPU", cpu_poly_divide<scalar_t>);