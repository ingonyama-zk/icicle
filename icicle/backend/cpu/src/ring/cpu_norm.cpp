#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/rings/integer_rings/labrador.h"  // For field_t definition
#include "taskflow/taskflow.hpp"
#include <cmath>
#include <atomic>

static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

// Get number of threads from config (defined in cpu_vec_ops.cpp)
int get_nof_workers(const VecOpsConfig& config);

typedef __uint128_t uint128_t;

static int64_t abs_centered(int64_t val, int64_t q)
{
  if (val > q / 2) { val = q - val; }
  return val;
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

  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (q < 0) {
    ICICLE_LOG_ERROR << "Field modulus q must be less than 64 bits; received q = " << q;
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Convert input to int64_t for processing
  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);

  // For L2 norm, we need to check sum of squares
  if (norm == eNormType::L2) {
    const uint64_t bound_squared = norm_bound * norm_bound;
    std::atomic<uint64_t> sum_squares(0);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const int nof_workers = get_nof_workers(config);
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers; // round up

    for (uint64_t start_idx = 0; start_idx < size; start_idx += worker_task_size) {
      taskflow.emplace([=, &sum_squares]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, size);
        uint128_t local_sum = 0;

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          // TODO: emirsoyturk validate input value is in the range [0, sqrt(q))
          int64_t val = abs_centered(input_i64[idx], q);
          local_sum += static_cast<uint128_t>(val) * static_cast<uint128_t>(val);
        }

        sum_squares.fetch_add(local_sum, std::memory_order_relaxed);
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

    *output = sum_squares.load(std::memory_order_relaxed) <= bound_squared;
  } 
  // For L-infinity norm, we just need to check the maximum absolute value
  else if (norm == eNormType::LInfinity) {
    std::atomic<int64_t> max_abs(0);

    tf::Taskflow taskflow;
    tf::Executor executor;
    const int nof_workers = get_nof_workers(config);
    const uint64_t worker_task_size = (size + nof_workers - 1) / nof_workers;

    for (uint64_t start_idx = 0; start_idx < size; start_idx += worker_task_size) {
      taskflow.emplace([=, &max_abs]() {
        const uint64_t end_idx = std::min(start_idx + worker_task_size, size);
        int64_t local_max = 0;

        for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
          int64_t val = abs_centered(input_i64[idx], q);
          local_max = std::max(local_max, val);
        }

        int64_t current_max = max_abs.load(std::memory_order_relaxed);
        if (local_max > current_max) {
          max_abs.compare_exchange_weak(current_max, local_max, 
                std::memory_order_relaxed, std::memory_order_relaxed);
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();

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
  // TODO: Implement relative norm check
  return eIcicleError::SUCCESS;
}

// Register the backend implementations
REGISTER_NORM_CHECK_BACKEND("CPU", cpu_check_norm_bound, cpu_check_norm_relative);
