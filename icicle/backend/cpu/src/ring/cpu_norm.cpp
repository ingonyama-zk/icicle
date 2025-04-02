#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "taskflow/taskflow.hpp"
#include <cmath>

typedef __uint128_t uint128_t;

static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

// Helper function to convert a value to centered representation [-q/2, q/2)
static inline int64_t abs_centered(int64_t val, int64_t q) {
  if (val <= q / 2) {
    return val;
  } else {
    return q - val;
  }
}

static inline uint128_t square(int64_t val, int64_t q) {
  return abs_centered(val, q) * abs_centered(val, q);
}

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
    uint128_t bound_squared = square(norm_bound, q);
    uint128_t norm_squared = 0;

    for (size_t i = 0; i < size; ++i) {
      int64_t val = input_i64[i];
      
      val = abs_centered(val, q);
      norm_squared += square(val, q);
      if (norm_squared >= bound_squared) {
        is_bounded = false;
        break;
      }
    }
  } else { // LInfinity norm
    // For LInfinity norm, we check if any element's absolute value exceeds the bound
    for (size_t i = 0; i < size; ++i) {
      int64_t val = input_i64[i];
      
      val = abs_centered(val, q);
      if (val >= norm_bound) {
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
    uint128_t scale_bound_squared = square(scale, q);
    uint128_t norm_a_squared = 0;
    uint128_t norm_b_squared = 0;

    for (size_t i = 0; i < size; ++i) {
      int64_t val_a = input_a_i64[i];
      int64_t val_b = input_b_i64[i];

      val_a = abs_centered(val_a, q);
      val_b = abs_centered(val_b, q);

      norm_a_squared += square(val_a, q);
      norm_b_squared += square(val_b, q);
    }

    // Check if norm_a^2 < (scale * norm_b)^2
    is_bounded = norm_a_squared < scale_bound_squared * norm_b_squared;
  } else { // LInfinity norm
    // For LInfinity norm, we check if any element's absolute value exceeds scale * bound
    for (size_t i = 0; i < size; ++i) {
      int64_t val_a = input_a_i64[i];
      int64_t val_b = input_b_i64[i];

      val_a = abs_centered(val_a, q);
      val_b = abs_centered(val_b, q);

      if (val_a >= scale * val_b) {
        is_bounded = false;
        break;
      }
    }
  }

  *output = is_bounded;
  return eIcicleError::SUCCESS;
}

REGISTER_NORM_CHECK_BACKEND("CPU", cpu_check_norm_bound, cpu_check_norm_relative);
