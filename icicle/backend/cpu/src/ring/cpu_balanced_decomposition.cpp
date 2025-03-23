#include "icicle/backend/vec_ops_backend.h"

static eIcicleError cpu_decompose_balanced_digits(
  const Device& device,
  const field_t* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  field_t* output,
  size_t output_size)
{
  static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const uint64_t q = *(uint64_t*)&q_storage; // Note this is valid since TLC == 2

  if (input == nullptr || output == nullptr || input_size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument";
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (base <= 1) {
    ICICLE_LOG_ERROR << "Invalid argument - base must be greater than 1";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const size_t digits_per_element = static_cast<size_t>(std::ceil(std::log2(q) / std::log2(base)));
  if (output_size < input_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Output buffer too small for decomposition";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Balanced decomposition does not support column batch";
    return eIcicleError::INVALID_ARGUMENT;
  }

  uint64_t stride = 1; // since column_batch is not supported
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    const field_t* curr_input = input + idx_in_batch * input_size;
    field_t* curr_output = output + idx_in_batch * input_size;
    for (uint64_t i = 0; i < input_size; ++i) {
      field_t val = curr_input[i * stride];
      // TODO: Implement balanced decomposition
      //       for (uint32_t j = 0; j < base; ++j) {
      //         curr_output[j * input_size + i * stride] = val % base;
      //         val /= base;
      //       }
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
  static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const uint64_t q = *(uint64_t*)&q_storage; // Note this is valid since TLC == 2

  if (input == nullptr || output == nullptr || input_size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument";
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (base <= 1) {
    ICICLE_LOG_ERROR << "Invalid argument - base must be greater than 1";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Balanced decomposition does not support column batch";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const size_t digits_per_element = static_cast<size_t>(std::ceil(std::log2(q) / std::log2(base)));

  if (output_size < input_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Output buffer too small for recomposition";
    return eIcicleError::INVALID_ARGUMENT;
  }

  uint64_t stride = 1; // since column_batch is not supported
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; ++idx_in_batch) {
    const field_t* curr_input = input + idx_in_batch * output_size;
    field_t* curr_output = output + idx_in_batch * output_size;
    for (uint64_t i = 0; i < output_size; ++i) {
      field_t val = field_t::zero();
      // TODO: Implement balanced decomposition
      //   for (uint32_t j = base; j > 0; --j) {
      //     val = val * base + curr_input[j * output_size + i * stride];
      //   }
      //   curr_output[i * stride] = val;
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_BALANCED_DECOMPOSITION_BACKEND("CPU", cpu_decompose_balanced_digits, cpu_recompose_from_balanced_digits);