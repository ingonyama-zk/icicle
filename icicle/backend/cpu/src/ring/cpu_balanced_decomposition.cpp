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
  const uint64_t q = *(const uint64_t*)&q_storage;

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

  const size_t digits_per_element =
    static_cast<size_t>(std::ceil(std::log2(static_cast<double>(q)) / std::log2(static_cast<double>(base))));

  if (output_size < input_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Output buffer too small for balanced decomposition.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    const field_t* curr_input = input + idx_in_batch * input_size;
    field_t* curr_output = output + idx_in_batch * input_size * digits_per_element;

    for (size_t i = 0; i < input_size; ++i) {
      int64_t element_to_decompose = *reinterpret_cast<const int64_t*>(curr_input + i);

      for (size_t j = 0; j < digits_per_element; ++j) {
        int64_t digit = element_to_decompose % base;
        element_to_decompose /= base;

        // Shift into balanced digit range [-b/2, b/2)
        if (digit > static_cast<int64_t>(base) / 2) {
          digit -= base;
          ++element_to_decompose;
        }

        int64_t* curr_output_int64 = reinterpret_cast<int64_t*>(curr_output + i * digits_per_element + j);
        *curr_output_int64 = (digit < 0) ? digit + q : digit; // TODO Yuval: is this correct?
      }
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
  const uint64_t q = *(const uint64_t*)&q_storage;

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

  const size_t digits_per_element =
    static_cast<size_t>(std::ceil(std::log2(static_cast<double>(q)) / std::log2(static_cast<double>(base))));

  if (input_size < output_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Input buffer too small for balanced recomposition.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    const int64_t* curr_input = reinterpret_cast<const int64_t*>(input) + idx_in_batch * input_size;
    int64_t* curr_output = reinterpret_cast<int64_t*>(output) + idx_in_batch * (input_size * digits_per_element);

    for (size_t out_idx = 0; out_idx < output_size; ++out_idx) {
      int64_t acc = 0;

      for (size_t j = digits_per_element; j-- > 0;) {
        acc = acc * base + *(curr_input + out_idx * digits_per_element + j);
      }

      curr_output[out_idx] = acc;
    }
  }

  return eIcicleError::SUCCESS;
}

REGISTER_BALANCED_DECOMPOSITION_BACKEND("CPU", cpu_decompose_balanced_digits, cpu_recompose_from_balanced_digits);