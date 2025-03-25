#include "icicle/balanced_decomposition.h"
#include "icicle/backend/vec_ops_backend.h"
#include <cmath>

static eIcicleError cpu_decompose_balanced_digits(
  const Device& device,
  const field_t* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  field_t* output,
  size_t output_size)
{
  static_assert(field_t::TLC == 2, "Balanced decomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;
  {
    ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit";
  }

  if (!input || !output || input_size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer or zero size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (base <= 1) {
    ICICLE_LOG_ERROR << "Invalid base: must be greater than 1.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Unsupported config: balanced decomposition does not support column batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const size_t digits_per_element = compute_nof_digits<field_t>(base);
  if (output_size < input_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Output buffer too small for balanced decomposition.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);
  int64_t* output_i64 = reinterpret_cast<int64_t*>(output);

  auto divmod = [](int64_t a, int64_t base) -> std::pair<int64_t, int64_t> {
    int64_t q = a / base;
    int64_t r = a % base;
    if ((r != 0) && ((a ^ base) < 0)) {
      q -= 1;
      r += base;
    }
    return {q, r};
  };

  // TODO: Replace with parallel task manager for performance.
  for (size_t idx = 0; idx < input_size * config.batch_size; ++idx) {
    int64_t val = input_i64[idx];
    int64_t digit = 0;
    // we need to handle case where val>q/2 by subtracting q (only for base>2)
    if (base > 2 && val > q / 2) { val = val - q; }

    for (size_t digit_idx = 0; digit_idx < digits_per_element; ++digit_idx) {
      std::tie(val, digit) = divmod(val, base);

      // Shift into balanced digit range [-b/2, b/2)
      if (digit > static_cast<int64_t>(base) / 2) {
        digit -= static_cast<int64_t>(base);
        ++val;
      }

      // Wrap negative digits to [0, q) for representation in field_t
      output_i64[idx * digits_per_element + digit_idx] = digit < 0 ? digit + q : digit;
    }
    if (val != 0) {
      ICICLE_LOG_ERROR << "Balanced decomposition failed: input value too large.";
      return eIcicleError::INVALID_ARGUMENT;
    }
  }

  return eIcicleError::SUCCESS;
}

static eIcicleError cpu_recompose_from_balanced_digits(
  const Device& device,
  const field_t* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  field_t* output,
  size_t output_size)
{
  static_assert(field_t::TLC == 2, "Balanced recomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;
  {
    ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit";
  }

  if (!input || !output || input_size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer or zero size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (base <= 1) {
    ICICLE_LOG_ERROR << "Invalid base: must be greater than 1.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Unsupported config: balanced decomposition does not support column batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const size_t digits_per_element = compute_nof_digits<field_t>(base);
  if (input_size < output_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Input buffer too small for balanced recomposition.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  // TODO: Replace with parallel task manager for performance.
  field_t base_as_field = field_t::from(base);
  for (size_t out_idx = 0; out_idx < output_size * config.batch_size; ++out_idx) {
    // TODO: can maybe implement with i64 type but not sure if faster
    field_t acc = field_t::zero();

    for (size_t digit_idx = digits_per_element; digit_idx-- > 0;) {
      auto digit = input[out_idx * digits_per_element + digit_idx];
      acc = acc * base_as_field + digit;
    }

    output[out_idx] = acc;
  }

  return eIcicleError::SUCCESS;
}

REGISTER_BALANCED_DECOMPOSITION_BACKEND("CPU", cpu_decompose_balanced_digits, cpu_recompose_from_balanced_digits);