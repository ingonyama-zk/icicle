#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "taskflow/taskflow.hpp"
#include <cmath>
#include <atomic>

static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

// Get number of threads from config (defined in cpu_vec_ops.cpp)
int get_nof_workers(const VecOpsConfig& config);

using uint128_t = __uint128_t;

static uint64_t abs_centered(uint64_t val, uint64_t q)
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
    ICICLE_LOG_ERROR << "Invalid pointer: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (size > 65536) { // size of the element shouldn't be bigger than 2^16
    ICICLE_LOG_ERROR << "Invalid argument: vector size must be at most 65536.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: vector size must be greater than 0.";
    return eIcicleError::INVALID_ARGUMENT;
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
    std::vector<std::vector<uint128_t>> thread_sums(config.batch_size, std::vector<uint128_t>(nof_workers, 0));
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0, thread_idx = 0; start_idx < size; start_idx += worker_task_size, ++thread_idx) {
      taskflow.emplace([=, &thread_sums, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        std::vector<uint128_t> local_sums(config.batch_size, 0);

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
            int64_t val = input_i64[batch_idx * size + idx];
            val = abs_centered(val, q);
            if (!validate_input_range(val, sqrt_q)) {
              validation_failed.store(true, std::memory_order_relaxed);
              return;
            }
            local_sums[batch_idx] += static_cast<uint128_t>(val) * static_cast<uint128_t>(val);
          }
        }

        for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
          thread_sums[batch_idx][thread_idx] = local_sums[batch_idx];
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
      uint128_t total_sum = 0;
      for (const auto& sum : thread_sums[batch_idx]) {
        total_sum += sum;
      }
      output[batch_idx] = total_sum < bound_squared;
    }
  }
  // For L-infinity norm, we just need to check the max(|input|) < norm_bound
  else if (norm == eNormType::LInfinity) {
    std::vector<std::atomic<int64_t>> max_abs(config.batch_size);
    for (auto& max : max_abs) {
      max.store(0, std::memory_order_relaxed);
    }
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const int nof_workers = get_nof_workers(config);
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0; start_idx < size; start_idx += worker_task_size) {
      taskflow.emplace([=, &max_abs, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        std::vector<int64_t> local_max(config.batch_size, 0);

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
            int64_t val = input_i64[batch_idx * size + idx];
            val = abs_centered(val, q);
            if (!validate_input_range(val, sqrt_q)) {
              validation_failed.store(true, std::memory_order_relaxed);
              return;
            }
            local_max[batch_idx] = std::max(local_max[batch_idx], val);
          }
        }

        for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
          int64_t current_max = max_abs[batch_idx].load(std::memory_order_relaxed);
          if (local_max[batch_idx] > current_max) {
            max_abs[batch_idx].compare_exchange_weak(
              current_max, local_max[batch_idx], std::memory_order_relaxed, std::memory_order_relaxed);
          }
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
      output[batch_idx] = max_abs[batch_idx].load(std::memory_order_relaxed) < static_cast<int64_t>(norm_bound);
    }
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
    ICICLE_LOG_ERROR << "Invalid pointer: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (size > 65536) { // size of the element shouldn't be bigger than 2^16
    ICICLE_LOG_ERROR << "Invalid argument: vector size must be at most 65536.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: vector size must be greater than 0.";
    return eIcicleError::INVALID_ARGUMENT;
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
    std::vector<std::vector<uint128_t>> thread_sums_a(config.batch_size, std::vector<uint128_t>(nof_workers, 0));
    std::vector<std::vector<uint128_t>> thread_sums_b(config.batch_size, std::vector<uint128_t>(nof_workers, 0));
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0, thread_idx = 0; start_idx < size; start_idx += worker_task_size, ++thread_idx) {
      taskflow.emplace([=, &thread_sums_a, &thread_sums_b, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        std::vector<uint128_t> local_sum_a(config.batch_size, 0);
        std::vector<uint128_t> local_sum_b(config.batch_size, 0);

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
            int64_t val_a = input_a_i64[batch_idx * size + idx];
            val_a = abs_centered(val_a, q);
            if (!validate_input_range(val_a, sqrt_q)) {
              validation_failed.store(true, std::memory_order_relaxed);
              return;
            }
            local_sum_a[batch_idx] += static_cast<uint128_t>(val_a) * static_cast<uint128_t>(val_a);

            int64_t val_b = input_b_i64[batch_idx * size + idx];
            val_b = abs_centered(val_b, q);
            if (!validate_input_range(val_b, sqrt_q)) {
              validation_failed.store(true, std::memory_order_relaxed);
              return;
            }
            local_sum_b[batch_idx] += static_cast<uint128_t>(val_b) * static_cast<uint128_t>(val_b);
          }
        }

        for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
          thread_sums_a[batch_idx][thread_idx] = local_sum_a[batch_idx];
          thread_sums_b[batch_idx][thread_idx] = local_sum_b[batch_idx];
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
      uint128_t norm_a_squared = 0;
      uint128_t norm_b_squared = 0;
      for (size_t i = 0; i < nof_workers; ++i) {
        norm_a_squared += thread_sums_a[batch_idx][i];
        norm_b_squared += thread_sums_b[batch_idx][i];
      }

      const uint128_t scale_squared = static_cast<uint128_t>(scale) * static_cast<uint128_t>(scale);
      output[batch_idx] = norm_a_squared < scale_squared * norm_b_squared;
    }
  }
  // For L-infinity norm, we need to check max(|input_a|) < scale * max(|input_b|)
  else if (norm == eNormType::LInfinity) {
    std::vector<std::atomic<int64_t>> max_abs_a(config.batch_size);
    std::vector<std::atomic<int64_t>> max_abs_b(config.batch_size);
    for (auto& max : max_abs_a) {
      max.store(0, std::memory_order_relaxed);
    }
    for (auto& max : max_abs_b) {
      max.store(0, std::memory_order_relaxed);
    }
    std::atomic<bool> validation_failed(false);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const int nof_workers = get_nof_workers(config);
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0; start_idx < size; start_idx += worker_task_size) {
      taskflow.emplace([=, &max_abs_a, &max_abs_b, &validation_failed]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, static_cast<uint64_t>(size));
        std::vector<int64_t> local_max_a(config.batch_size, 0);
        std::vector<int64_t> local_max_b(config.batch_size, 0);

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
            int64_t val_a = input_a_i64[batch_idx * size + idx];
            val_a = abs_centered(val_a, q);
            if (!validate_input_range(val_a, sqrt_q)) {
              validation_failed.store(true, std::memory_order_relaxed);
              return;
            }
            local_max_a[batch_idx] = std::max(local_max_a[batch_idx], val_a);

            int64_t val_b = input_b_i64[batch_idx * size + idx];
            val_b = abs_centered(val_b, q);
            if (!validate_input_range(val_b, sqrt_q)) {
              validation_failed.store(true, std::memory_order_relaxed);
              return;
            }
            local_max_b[batch_idx] = std::max(local_max_b[batch_idx], val_b);
          }
        }

        for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
          int64_t current_max_a = max_abs_a[batch_idx].load(std::memory_order_relaxed);
          while (local_max_a[batch_idx] > current_max_a) {
            if (max_abs_a[batch_idx].compare_exchange_weak(
                  current_max_a, local_max_a[batch_idx], std::memory_order_relaxed, std::memory_order_relaxed)) {
              break;
            }
          }

          int64_t current_max_b = max_abs_b[batch_idx].load(std::memory_order_relaxed);
          while (local_max_b[batch_idx] > current_max_b) {
            if (max_abs_b[batch_idx].compare_exchange_weak(
                  current_max_b, local_max_b[batch_idx], std::memory_order_relaxed, std::memory_order_relaxed)) {
              break;
            }
          }
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    if (validation_failed.load(std::memory_order_relaxed)) { return eIcicleError::INVALID_ARGUMENT; }

    for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
      const int64_t norm_a = max_abs_a[batch_idx].load(std::memory_order_relaxed);
      const int64_t norm_b = max_abs_b[batch_idx].load(std::memory_order_relaxed);

      output[batch_idx] = norm_a < static_cast<int64_t>(scale) * norm_b;
    }
  }

  return eIcicleError::SUCCESS;
}

// Register the backend implementations
REGISTER_NORM_CHECK_BACKEND("CPU", cpu_check_norm_bound, cpu_check_norm_relative);
