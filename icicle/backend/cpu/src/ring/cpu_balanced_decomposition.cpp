#include "icicle/balanced_decomposition.h"
#include "icicle/backend/vec_ops_backend.h"
#include "taskflow/taskflow.hpp"
#include <cmath>
#include <cstdint>
#include <taskflow/core/executor.hpp>
#include <taskflow/core/taskflow.hpp>

static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

// Helper function that performs floor division and modulo like Python or standard math.
// Ensures that the remainder is always non-negative and the quotient is rounded down
// (i.e., toward negative infinity), unlike C++'s default behavior which rounds toward zero.
static inline std::pair<int64_t, int64_t> divmod(int64_t a, uint32_t base)
{
  // Perform regular C++ integer division and modulo
  int64_t q = a / base;
  int64_t r = a % base;

  // If remainder is non-zero AND a and base have opposite signs
  // then C++ has rounded the quotient toward zero instead of toward -∞,
  // and the remainder is negative. Fix it.
  //
  // (a ^ base) < 0 checks if the signs of a and base are different:
  // - XOR of two values with the same sign yields a positive or zero result
  // - XOR of two values with different signs yields a negative result
  if ((r != 0) && ((a ^ base) < 0)) {
    q -= 1;    // Round quotient down by one
    r += base; // Adjust remainder to stay consistent with a = q * base + r
  }

  return {q, r};
}

template <typename T>
int64_t get_q()
{
  constexpr auto q_storage = T::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;
  return q;
}

template <typename T>
static eIcicleError verify_params(
  const T* input, size_t input_size, uint32_t base, const VecOpsConfig& config, T* output, size_t output_size)
{
  static_assert(field_t::TLC == 2, "Balanced decomposition assumes q ~64b");
  auto q = get_q<T>();

  if (q < 0) {
    ICICLE_LOG_ERROR << "Field modulus q must be less than 64 bits; received q = " << q;
    return eIcicleError::INVALID_ARGUMENT;
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
  return eIcicleError::SUCCESS;
}

// Decomposes Zq elements into balanced base-b digits.
static eIcicleError cpu_decompose_balanced_digits(
  const Device& device,
  const field_t* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  field_t* output,
  size_t output_size)
{
  auto params_valid = verify_params<field_t>(input, input_size, base, config, output, output_size);
  if (eIcicleError::SUCCESS != params_valid) { return params_valid; }

  auto q = get_q<field_t>();

  const size_t digits_per_element = balanced_decomposition::compute_nof_digits<field_t>(base);
  if (output_size < input_size * digits_per_element) {
    ICICLE_LOG_ERROR << "Output buffer too small for balanced decomposition.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);
  int64_t* output_i64 = reinterpret_cast<int64_t*>(output);

  const auto base_div2 = base / 2;
  const auto q_div2 = q / 2;

  tf::Taskflow taskflow;
  tf::Executor executor;
  const uint64_t total_nof_operations = input_size * config.batch_size;
  const int nof_workers = get_nof_workers(config);
  const uint64_t worker_task_size = (total_nof_operations + nof_workers - 1) / nof_workers; // round up

  std::atomic<bool> error = false;
  for (uint64_t start_idx = 0; start_idx < total_nof_operations; start_idx += worker_task_size) {
    taskflow.emplace([=, &error]() {
      const uint64_t end_idx = std::min(start_idx + worker_task_size, total_nof_operations);
      for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
        int64_t val = input_i64[idx];
        int64_t digit = 0;
        // we need to handle case where val>q/2 by subtracting q (only for base>2)
        if (base > 2 && val > q_div2) { val = val - q; }

        for (size_t digit_idx = 0; digit_idx < digits_per_element; ++digit_idx) {
          std::tie(val, digit) = divmod(val, base);

          // Shift into balanced digit range [-b/2, b/2)
          if (digit > base_div2) {
            digit -= base;
            ++val;
          }

          // Wrap negative digits to [0, q) for representation in field_t
          output_i64[idx * digits_per_element + digit_idx] = digit < 0 ? digit + q : digit;
        }
        if (val != 0) {
          if (error) { return; } // stop processing if another thread already failed
          ICICLE_LOG_ERROR << "Balanced decomposition failed: input value too large.";
          error.store(true, std::memory_order_relaxed); // mark failure
          return;
        }
      }
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();

  return error ? eIcicleError::INVALID_ARGUMENT : eIcicleError::SUCCESS;
}

// Recompose Zq elements from balanced base-b digits.
static eIcicleError cpu_recompose_from_balanced_digits(
  const Device& device,
  const field_t* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  field_t* output,
  size_t output_size)
{
  auto params_valid = verify_params<field_t>(input, input_size, base, config, output, output_size);
  if (eIcicleError::SUCCESS != params_valid) { return params_valid; }

  const size_t digits_per_element = input_size / output_size;
  if (input_size % output_size != 0) {
    ICICLE_LOG_ERROR << "Balanced recomposition: output size must divide input size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  field_t base_as_field = field_t::from(base);

  tf::Taskflow taskflow;
  tf::Executor executor;
  const uint64_t total_nof_operations = output_size * config.batch_size;
  const int nof_workers = get_nof_workers(config);
  const uint64_t worker_task_size = (total_nof_operations + nof_workers - 1) / nof_workers; // round up

  for (uint64_t start_idx = 0; start_idx < total_nof_operations; start_idx += worker_task_size) {
    taskflow.emplace([=]() {
      const uint64_t end_idx = std::min(start_idx + worker_task_size, total_nof_operations);
      for (uint64_t out_idx = start_idx; out_idx < end_idx; ++out_idx) {
        field_t acc = field_t::zero();
        // computing 'x ≡ ∑ r_i * b^i mod q' in field_t
        for (int digit_idx = digits_per_element - 1; digit_idx >= 0; --digit_idx) {
          auto digit = input[out_idx * digits_per_element + digit_idx];
          acc = acc * base_as_field + digit;
        }
        output[out_idx] = acc;
      }
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();

  return eIcicleError::SUCCESS;
}

REGISTER_BALANCED_DECOMPOSITION_BACKEND("CPU", cpu_decompose_balanced_digits, cpu_recompose_from_balanced_digits);

// Decompose PolyRing polynomials into balanced base-b digits
static eIcicleError cpu_decompose_balanced_digits_PolyRing(
  const Device& device,
  const PolyRing* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  PolyRing* output,
  size_t output_size)
{
  auto params_valid = verify_params<PolyRing::Base>(
    (PolyRing::Base*)input, input_size, base, config, (PolyRing::Base*)output, output_size);
  if (eIcicleError::SUCCESS != params_valid) { return params_valid; }

  auto q = get_q<PolyRing::Base>();

  // Check that the sizes make sense and that we have enough digits.
  // Note that not enough digits might be an issue but treated as warning.
  const size_t digits_per_element = output_size / input_size;
  if (output_size % input_size != 0) {
    ICICLE_LOG_ERROR << "Balanced decomposition: output size must divide input size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const size_t expected_digits_per_element = balanced_decomposition::compute_nof_digits<PolyRing::Base>(base);
  if (digits_per_element < expected_digits_per_element) {
    ICICLE_LOG_WARNING << "Balanced Decomposition: Output buffer may be too small to decompose input polynomials. "
                          "Decomposition will stop after "
                       << digits_per_element << "digits, based on output size.";
  }

  const auto base_div2 = base / 2;
  const auto q_div2 = q / 2;

  tf::Taskflow tasks;
  tf::Executor executor(get_nof_workers(config));

  // To decompose PolyRing polynomials into balanced digits, we decompose all d coefficients, in num_digits steps,
  // creating a polynomial on each step.
  const size_t total_size = input_size * config.batch_size;
  for (int poly_idx = 0; poly_idx < total_size; ++poly_idx) {
    tasks.emplace([=] {
      const PolyRing& input_poly = input[poly_idx];
      const int64_t* input_coeffs = reinterpret_cast<const int64_t*>(input_poly.values);

      // Temporary buffer to hold intermediate remainders during decomposition.
      int64_t remainder[PolyRing::d];
      std::memcpy(remainder, input_coeffs, sizeof(remainder));

      for (int digit_idx = 0; digit_idx < digits_per_element; ++digit_idx) {
        PolyRing& output_poly = output[digit_idx * total_size + poly_idx];
        int64_t* output_coeffs = reinterpret_cast<int64_t*>(output_poly.values);

        for (int coeff_idx = 0; coeff_idx < PolyRing::d; ++coeff_idx) {
          int64_t val = remainder[coeff_idx];
          int64_t digit = 0;

          // we need to handle case where val>q/2 by subtracting q (only for base>2)
          if (base > 2 && val > q_div2) { val -= q; }

          std::tie(val, digit) = divmod(val, base);
          // Shift into balanced digit range [-b/2, b/2)
          if (digit > base_div2) {
            digit -= base;
            ++val;
          }

          remainder[coeff_idx] = val;
          output_coeffs[coeff_idx] = digit < 0 ? digit + q : digit;
        }
      }
    });
  }

  executor.run(tasks).wait();
  tasks.clear();

  return eIcicleError::SUCCESS;
}

// Recompose PolyRing polynomials from balanced base-b digits
static eIcicleError cpu_recompose_from_balanced_digits_PolyRing(
  const Device& device,
  const PolyRing* input,
  size_t input_size,
  uint32_t base,
  const VecOpsConfig& config,
  PolyRing* output,
  size_t output_size)
{
  auto params_valid = verify_params<PolyRing::Base>(
    (PolyRing::Base*)input, input_size, base, config, (PolyRing::Base*)output, output_size);
  if (eIcicleError::SUCCESS != params_valid) { return params_valid; }

  const size_t digits_per_element = input_size / output_size;
  if (input_size % output_size != 0) {
    ICICLE_LOG_ERROR << "Balanced recomposition: output size must divide input size.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  const PolyRing::Base base_as_field = PolyRing::Base::from(base);

  tf::Taskflow tasks;
  tf::Executor executor(get_nof_workers(config));

  const size_t total_size = output_size * config.batch_size;
  for (int poly_idx = 0; poly_idx < total_size; ++poly_idx) {
    tasks.emplace([=]() {
      PolyRing result;
      // Recompose a single output polynomial from its digit-wise components
      for (int coeff_idx = 0; coeff_idx < PolyRing::d; ++coeff_idx) {
        PolyRing::Base acc = PolyRing::Base::zero();
        // Accumulate from most significant digit to least
        for (int digit_idx = digits_per_element - 1; digit_idx >= 0; --digit_idx) {
          const PolyRing& input_poly = input[digit_idx * total_size + poly_idx];
          acc = acc * base_as_field + input_poly.values[coeff_idx];
        }
        result.values[coeff_idx] = acc;
      }
      output[poly_idx] = result;
    });
  }

  executor.run(tasks).wait();
  tasks.clear();

  return eIcicleError::SUCCESS;
}

REGISTER_BALANCED_DECOMPOSITION_POLYRING_BACKEND(
  "CPU", cpu_decompose_balanced_digits_PolyRing, cpu_recompose_from_balanced_digits_PolyRing);
