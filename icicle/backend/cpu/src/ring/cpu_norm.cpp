#include "icicle/norm.h"
#include "icicle/backend/vec_ops_backend.h"
#include "taskflow/taskflow.hpp"
#include <cmath>
#include <atomic>

static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");

// Get number of threads from config (defined in cpu_vec_ops.cpp)
int get_nof_workers(const VecOpsConfig& config);

static int64_t abs_centered(int64_t val, int64_t q)
{
  if (val > q / 2) { val = q - val; }
  return val;
}

// validate input elements don't exceed √q
static bool validate_inputs(const field_t* input, size_t size, int64_t q)
{
  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);
  for (size_t i = 0; i < size; ++i) {
    if (input_i64[i] * input_i64[i] >= q) {
      ICICLE_LOG_ERROR << "Input element exceeds field modulus q";
      return false;
    }
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
  if (!input || !output || size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer or zero size";
    return eIcicleError::INVALID_ARGUMENT;
  }

  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (!validate_inputs(input, size * config.batch_size, q)) { return eIcicleError::INVALID_ARGUMENT; }

  const int64_t* input_i64 = reinterpret_cast<const int64_t*>(input);
  *output = true;

  tf::Taskflow taskflow;
  tf::Executor executor;
  const uint64_t total_elements = size * config.batch_size;
  const int nof_workers = get_nof_workers(config);
  const uint64_t worker_task_size = (total_elements + nof_workers - 1) / nof_workers;

  std::atomic<bool> early_exit{false};

  if (norm == eNormType::L2) {
    std::vector<__int128> partial_sums(nof_workers, 0);
    const __int128 bound_squared = static_cast<__int128>(norm_bound) * norm_bound;
    __int128 total_sum = 0;

    for (uint64_t worker_id = 0; worker_id < nof_workers; worker_id++) {
      taskflow.emplace([=, &early_exit, &total_sum]() {
        if (early_exit.load(std::memory_order_relaxed)) return;

        const uint64_t start_idx = worker_id * worker_task_size;
        const uint64_t end_idx = std::min(start_idx + worker_task_size, total_elements);

        for (uint64_t i = start_idx; i < end_idx; ++i) {
          int64_t abs_val = abs_centered(input_i64[i], q);
          __int128 val_squared = static_cast<__int128>(abs_val) * abs_val;
          total_sum += val_squared;
          if (total_sum > bound_squared) {
            *output = false;
            early_exit.store(true, std::memory_order_relaxed);
            return;
          }
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();
  } else { // L∞ norm
    for (uint64_t start_idx = 0; start_idx < total_elements; start_idx += worker_task_size) {
      taskflow.emplace([=, &early_exit]() {
        if (early_exit.load(std::memory_order_relaxed)) return;

        const uint64_t end_idx = std::min(start_idx + worker_task_size, total_elements);

        for (uint64_t i = start_idx; i < end_idx; ++i) {
          int64_t abs_val = abs_centered(input_i64[i], q);
          if (abs_val >= norm_bound) {
            *output = false;
            early_exit.store(true, std::memory_order_relaxed);
            return;
          }
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();
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
  if (!input_a || !input_b || !output || size == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer or zero size";
    return eIcicleError::INVALID_ARGUMENT;
  }

  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;

  if (
    !validate_inputs(input_a, size * config.batch_size, q) || !validate_inputs(input_b, size * config.batch_size, q)) {
    return eIcicleError::INVALID_ARGUMENT;
  }

  const int64_t* input_a_i64 = reinterpret_cast<const int64_t*>(input_a);
  const int64_t* input_b_i64 = reinterpret_cast<const int64_t*>(input_b);
  *output = true;

  tf::Taskflow taskflow;
  tf::Executor executor;
  const uint64_t total_elements = size * config.batch_size;
  const int nof_workers = get_nof_workers(config);
  const uint64_t worker_task_size = (total_elements + nof_workers - 1) / nof_workers;

  std::atomic<bool> early_exit{false};

  if (norm == eNormType::L2) {
    __int128 total_sum_a = 0;
    __int128 total_sum_b = 0;

    for (uint64_t worker_id = 0; worker_id < nof_workers; worker_id++) {
      taskflow.emplace([=, &early_exit, &total_sum_a, &total_sum_b]() {
        if (early_exit.load(std::memory_order_relaxed)) return;

        const uint64_t start_idx = worker_id * worker_task_size;
        const uint64_t end_idx = std::min(start_idx + worker_task_size, total_elements);

        for (uint64_t i = start_idx; i < end_idx; ++i) {
          int64_t abs_a = abs_centered(input_a_i64[i], q);
          int64_t abs_b = abs_centered(input_b_i64[i], q);

          __int128 val_a_squared = static_cast<__int128>(abs_a) * abs_a;
          __int128 val_b_squared = static_cast<__int128>(abs_b) * abs_b;
          total_sum_a += val_a_squared;
          total_sum_b += val_b_squared;

          // ||a|| >= scale * ||b||
          if (total_sum_a >= static_cast<__int128>(scale) * total_sum_b) {
            *output = false;
            early_exit.store(true, std::memory_order_relaxed);
            return;
          }
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();
  } else { // L∞ norm
    for (uint64_t start_idx = 0; start_idx < total_elements; start_idx += worker_task_size) {
      taskflow.emplace([=, &early_exit]() {
        if (early_exit.load(std::memory_order_relaxed)) return;

        const uint64_t end_idx = std::min(start_idx + worker_task_size, total_elements);

        for (uint64_t i = start_idx; i < end_idx; ++i) {
          int64_t abs_a = abs_centered(input_a_i64[i], q);
          int64_t abs_b = abs_centered(input_b_i64[i], q);

          // |a| >= scale * |b|
          if (abs_a >= scale * abs_b) {
            *output = false;
            early_exit.store(true, std::memory_order_relaxed);
            return;
          }
        }
      });
    }
  }

  executor.run(taskflow).wait();
  taskflow.clear();

  return eIcicleError::SUCCESS;
}

// Register the backend implementations
REGISTER_NORM_CHECK_BACKEND("CPU", cpu_check_norm_bound, cpu_check_norm_relative);
