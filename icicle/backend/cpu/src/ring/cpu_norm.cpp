#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "taskflow/taskflow.hpp"
#include <cmath>

static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

// CPU implementation for icicle::norm::check_norm_bound()
static eIcicleError cpu_check_norm_bound(
  const Device& device,
  const field_t* input,
  size_t size,
  eNormType norm,
  uint64_t norm_bound,
  const VecOpsConfig& config,
  bool* output)
{
  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (q < 0) {
    ICICLE_LOG_ERROR << "Field modulus q must be less than 64 bits; received q = " << q;
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (!input || !output || size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer or zero size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Unsupported config: norm checking does not support column batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);
  bool is_bounded = true;

  if (norm == eNormType::L2) {
    // For L2 norm, we compute sum of squares and compare with bound^2
    // We use field_t to handle the intermediate calculations
    field_t bound_squared = field_t::from(norm_bound) * field_t::from(norm_bound);
    field_t norm_squared = field_t::from(0);

    for (size_t i = 0; i < size; ++i) {
      int64_t val = input_i64[i];
      // Convert to centered representation [-q/2, q/2)
      if (val > q/2) {
        val -= q;
      } else if (val < -q/2) {
        val += q;
      }
      norm_squared = norm_squared + field_t::from(val) * field_t::from(val);
      if (!field_t::lt(norm_squared, bound_squared)) {
        is_bounded = false;
        break;
      }
    }
  } else { // LInfinity norm
    // For LInfinity norm, we check if any element's absolute value exceeds the bound
    for (size_t i = 0; i < size; ++i) {
      int64_t val = input_i64[i];
      // Convert to centered representation [-q/2, q/2)
      if (val > q/2) { // TODO: emirsoyturk make this a function
        val -= q;
      } else if (val < -q/2) {
        val += q;
      }
      if (std::abs(val) >= static_cast<int64_t>(norm_bound)) {
        is_bounded = false;
        break;
      }
    }
  }

  *output = is_bounded;
  return eIcicleError::SUCCESS;
}

// CPU implementation for icicle::norm::check_norm_relative()
static eIcicleError cpu_check_norm_relative(
  const Device& device,
  const field_t* input_a,
  const field_t* input_b,
  size_t size,
  eNormType norm,
  uint64_t scale,
  const VecOpsConfig& config,
  bool* output)
{
  static_assert(field_t::TLC == 2, "Relative norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (q < 0) {
    ICICLE_LOG_ERROR << "Field modulus q must be less than 64 bits; received q = " << q;
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (!input_a || !input_b || !output || size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer or zero size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Unsupported config: relative norm checking does not support column batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const int64_t* input_a_i64 = reinterpret_cast<const int64_t*>(input_a);
  const int64_t* input_b_i64 = reinterpret_cast<const int64_t*>(input_b);
  bool is_bounded = true;

  if (norm == eNormType::L2) {
    // For L2 norm, we compute sum of squares and compare with (scale * bound)^2
    field_t scale_bound_squared = field_t::from(scale) * field_t::from(scale);
    field_t norm_a_squared = field_t::from(0);
    field_t norm_b_squared = field_t::from(0);

    for (size_t i = 0; i < size; ++i) {
      int64_t val_a = input_a_i64[i];
      int64_t val_b = input_b_i64[i];
      // Convert to centered representation [-q/2, q/2)
      if (val_a > q/2) val_a -= q;
      else if (val_a < -q/2) val_a += q;
      if (val_b > q/2) val_b -= q;
      else if (val_b < -q/2) val_b += q;

      norm_a_squared = norm_a_squared + field_t::from(val_a) * field_t::from(val_a);
      norm_b_squared = norm_b_squared + field_t::from(val_b) * field_t::from(val_b);
    }

    // Check if norm_a^2 < (scale * norm_b)^2
    is_bounded = field_t::lt(norm_a_squared, scale_bound_squared * norm_b_squared);
  } else { // LInfinity norm
    // For LInfinity norm, we check if any element's absolute value exceeds scale * bound
    for (size_t i = 0; i < size; ++i) {
      int64_t val_a = input_a_i64[i];
      int64_t val_b = input_b_i64[i];
      // Convert to centered representation [-q/2, q/2)
      if (val_a > q/2) val_a -= q;
      else if (val_a < -q/2) val_a += q;
      if (val_b > q/2) val_b -= q;
      else if (val_b < -q/2) val_b += q;

      if (std::abs(val_a) >= static_cast<int64_t>(scale) * std::abs(val_b)) {
        is_bounded = false;
        break;
      }
    }
  }

  *output = is_bounded;
  return eIcicleError::SUCCESS;
}

REGISTER_NORM_CHECK_BACKEND("CPU", cpu_check_norm_bound, cpu_check_norm_relative);
