#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "taskflow/taskflow.hpp"
#include <cmath>
#include <atomic>

static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

// Get number of threads from config (defined in cpu_vec_ops.cpp)
int get_nof_workers(const VecOpsConfig& config);

using uint128_t = __uint128_t;

static int64_t abs_centered(int64_t val, int64_t q)
{
  if (val > q / 2) { val = q - val; }
  return val;
}

static bool validate_input_range(int64_t val, int64_t sqrt_q)
{
  if (val >= sqrt_q) {
    ICICLE_LOG_ERROR << "Input value " << val << " is greater than sqrt(q) = " << sqrt_q;
    return false;
  }
  return true;
}

static eIcicleError cpu_check_norm_bound(
  const Device& device,
  const field_t* input,
  size_t size,
  eNormType norm,
  uint64_t norm_bound,
  const VecOpsConfig& config,
  bool* output)
{
  if (!input || !output) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (size == 0) {
    *output = true;
    return eIcicleError::SUCCESS;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Unsupported config: norm check does not support column batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (q < 0) {
    ICICLE_LOG_ERROR << "Field modulus q must be less than 64 bits; received q = " << q;
    return eIcicleError::INVALID_ARGUMENT;
  }

  auto sqrt_q = static_cast<uint32_t>(std::sqrt(q));

  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);

  // For L2 norm, we need to check the ||input||² < norm_bound²
  if (norm == eNormType::L2) {
    const uint128_t bound_squared = static_cast<uint128_t>(norm_bound) * static_cast<uint128_t>(norm_bound);
    const int nof_workers = get_nof_workers(config);
    std::vector<uint128_t> thread_sums(nof_workers, 0);
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0, thread_idx = 0; start_idx < size; start_idx += worker_task_size, ++thread_idx) {
      taskflow.emplace([=, &thread_sums, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        uint128_t local_sum = 0;

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          int64_t val = input_i64[idx];
          if (!validate_input_range(val, sqrt_q)) {
            validation_failed.store(true, std::memory_order_relaxed);
            return;
          }
          val = abs_centered(val, q);
          local_sum += static_cast<uint128_t>(val) * static_cast<uint128_t>(val);
        }

        thread_sums[thread_idx] = local_sum;
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }


    uint128_t total_sum = 0;
    for (const auto& sum : thread_sums) {
      total_sum += sum;
    }

    *output = total_sum <= bound_squared;
  }
  // For L-infinity norm, we just need to check the max(|input|) < norm_bound
  else if (norm == eNormType::LInfinity) {
    std::atomic<int64_t> max_abs(0);
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const int nof_workers = get_nof_workers(config);
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0; start_idx < size; start_idx += worker_task_size) {
      taskflow.emplace([=, &max_abs, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        int64_t local_max = 0;

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          int64_t val = input_i64[idx];
          if (!validate_input_range(val, sqrt_q)) {
            validation_failed.store(true, std::memory_order_relaxed);
            return;
          }
          val = abs_centered(val, q);
          local_max = std::max(local_max, val);
        }

        int64_t current_max = max_abs.load(std::memory_order_relaxed);
        if (local_max > current_max) {
          max_abs.compare_exchange_weak(current_max, local_max, std::memory_order_relaxed, std::memory_order_relaxed);
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    *output = max_abs.load(std::memory_order_relaxed) <= static_cast<int64_t>(norm_bound);
  }

  return eIcicleError::SUCCESS;
}

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
  if (!input_a || !input_b || !output) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (size == 0) {
    *output = true;
    return eIcicleError::SUCCESS;
  }

  if (config.columns_batch) {
    ICICLE_LOG_ERROR << "Unsupported config: norm check does not support column batch.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (q < 0) {
    ICICLE_LOG_ERROR << "Field modulus q must be less than 64 bits; received q = " << q;
    return eIcicleError::INVALID_ARGUMENT;
  }

  auto sqrt_q = static_cast<uint32_t>(std::sqrt(q));

  const int64_t* input_a_i64 = reinterpret_cast<const int64_t*>(input_a);
  const int64_t* input_b_i64 = reinterpret_cast<const int64_t*>(input_b);

  // For L2 norm, we need to check ||input_a||² < scale² * ||input_b||²
  if (norm == eNormType::L2) {
    const int nof_workers = get_nof_workers(config);
    std::vector<uint128_t> thread_sums_a(nof_workers, 0);
    std::vector<uint128_t> thread_sums_b(nof_workers, 0);
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0, thread_idx = 0; start_idx < size; start_idx += worker_task_size, ++thread_idx) {
      taskflow.emplace([=, &thread_sums_a, &thread_sums_b, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        uint128_t local_sum_a = 0;
        uint128_t local_sum_b = 0;

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          int64_t val_a = input_a_i64[idx];
          if (!validate_input_range(val_a, sqrt_q)) {
            validation_failed.store(true, std::memory_order_relaxed);
            return;
          }
          val_a = abs_centered(val_a, q);
          local_sum_a += static_cast<uint128_t>(val_a) * static_cast<uint128_t>(val_a);

          int64_t val_b = input_b_i64[idx];
          if (!validate_input_range(val_b, sqrt_q)) {
            validation_failed.store(true, std::memory_order_relaxed);
            return;
          }
          val_b = abs_centered(val_b, q);
          local_sum_b += static_cast<uint128_t>(val_b) * static_cast<uint128_t>(val_b);
        }

        thread_sums_a[thread_idx] = local_sum_a;
        thread_sums_b[thread_idx] = local_sum_b;
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    // Sum up all thread results in the main thread
    uint128_t norm_a_squared = 0;
    uint128_t norm_b_squared = 0;
    for (size_t i = 0; i < nof_workers; ++i) {
      norm_a_squared += thread_sums_a[i];
      norm_b_squared += thread_sums_b[i];
    }

    const uint128_t scale_squared = static_cast<uint128_t>(scale) * static_cast<uint128_t>(scale);
    *output = norm_a_squared < scale_squared * norm_b_squared;
  }
  // For L-infinity norm, we need to check max(|input_a|) < scale * max(|input_b|)
  else if (norm == eNormType::LInfinity) {
    std::atomic<int64_t> max_abs_a(0);
    std::atomic<int64_t> max_abs_b(0);
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const int nof_workers = get_nof_workers(config);
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0; start_idx < size; start_idx += worker_task_size) {
      taskflow.emplace([=, &max_abs_a, &max_abs_b, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        int64_t local_max_a = 0;
        int64_t local_max_b = 0;

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          int64_t val_a = input_a_i64[idx];
          if (!validate_input_range(val_a, sqrt_q)) {
            validation_failed.store(true, std::memory_order_relaxed);
            return;
          }
          val_a = abs_centered(val_a, q);
          local_max_a = std::max(local_max_a, val_a);

          int64_t val_b = input_b_i64[idx];
          if (!validate_input_range(val_b, sqrt_q)) {
            validation_failed.store(true, std::memory_order_relaxed);
            return;
          }
          val_b = abs_centered(val_b, q);
          local_max_b = std::max(local_max_b, val_b);
        }

        int64_t current_max_a = max_abs_a.load(std::memory_order_relaxed);
        while (local_max_a > current_max_a) {
          if (max_abs_a.compare_exchange_weak(
                current_max_a, local_max_a, std::memory_order_relaxed, std::memory_order_relaxed)) {
            break;
          }
        }

        int64_t current_max_b = max_abs_b.load(std::memory_order_relaxed);
        while (local_max_b > current_max_b) {
          if (max_abs_b.compare_exchange_weak(
                current_max_b, local_max_b, std::memory_order_relaxed, std::memory_order_relaxed)) {
            break;
          }
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    const int64_t norm_a = max_abs_a.load(std::memory_order_relaxed);
    const int64_t norm_b = max_abs_b.load(std::memory_order_relaxed);

    *output = norm_a < static_cast<int64_t>(scale) * norm_b;
  }

  return eIcicleError::SUCCESS;
}

// Register the backend implementations
REGISTER_NORM_CHECK_BACKEND("CPU", cpu_check_norm_bound, cpu_check_norm_relative);
