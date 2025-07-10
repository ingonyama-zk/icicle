#include "icicle/backend/vec_ops_backend.h"
#include "icicle/backend/mat_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "icicle/mat_ops.h"
#include "taskflow/taskflow.hpp"
#include <cmath>
#include <cstdint>
#include <taskflow/core/executor.hpp>
#include <taskflow/core/task.hpp>
#include <taskflow/core/taskflow.hpp>

// Extract number of threads to run from configuration
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

// Extract number of threads to run from MatmulConfig
int get_nof_workers(const MatMulConfig& config)
{
  if (config.ext && config.ext->has("n_threads")) { return config.ext->get<int>("n_threads"); }
  const int hw_threads = std::thread::hardware_concurrency();
  return std::max(1, hw_threads);
}

namespace {
  using namespace field_config;
  using namespace icicle;

  /**
   * @brief Generic CPU matrix multiplication.
   *
   * Each logical matrix element is a vector of `degree` field elements (e.g., a polynomial in evaluation form).
   * Computes: mat_out = mat_a × mat_b
   *
   * Template Parameters:
   * - Zq: the base field type
   * - degree: number of coefficients per matrix element (e.g., PolyRing<Zq, d> has degree = d)
   */
  template <typename T, uint32_t degree>
  static eIcicleError cpu_matmul_internal(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out)
  {
    if (!mat_a || !mat_b || !mat_out || nof_rows_a == 0 || nof_cols_a == 0 || nof_rows_b == 0 || nof_cols_b == 0) {
      ICICLE_LOG_ERROR << "Matmul: invalid size or nullptr input";
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (config.batch_size != 1) {
      ICICLE_LOG_ERROR << "Matmul does not support batching (batch_size > 1)";
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (nof_cols_a != nof_rows_b) {
      ICICLE_LOG_ERROR << "Matmul: inner dimensions do not match (cols_a != rows_b)";
      return eIcicleError::INVALID_ARGUMENT;
    }

    const int nof_workers = get_nof_workers(config);

    tf::Taskflow taskflow;
    tf::Executor executor(nof_workers);

    // One task per output row
    for (uint32_t row = 0; row < nof_rows_a; ++row) {
      taskflow.emplace([=]() {
        for (uint32_t col = 0; col < nof_cols_b; ++col) {
          T acc[degree];
          std::memset(acc, 0, sizeof(acc));

          // Compute dot product of row from A and column from B
          for (uint32_t k = 0; k < nof_cols_a; ++k) {
            const scalar_t* a = mat_a + (row * nof_cols_a + k) * degree;
            const scalar_t* b = mat_b + (k * nof_cols_b + col) * degree;
            for (uint32_t d = 0; d < degree; ++d) {
              acc[d] = acc[d] + a[d] * b[d];
            }
          }

          scalar_t* out = mat_out + (row * nof_cols_b + col) * degree;
          std::memcpy(out, acc, sizeof(T) * degree);
        }
      });
    }

    executor.run(taskflow).wait();
    taskflow.clear();
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Matrix multiplication for PolyRing<T::Base, T::d>
   *
   * Internally expands each ring element into its base field representation,
   * performs coefficient-wise multiplication, and reassembles the result.
   */
  template <typename T>
  static eIcicleError cpu_matmul_polynomial_ring(
    const Device& device,
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out)
  {
    using Zq = typename T::Base;
    constexpr uint32_t d = T::d;

    return cpu_matmul_internal<Zq, d>(
      reinterpret_cast<const Zq*>(mat_a), nof_rows_a, nof_cols_a, reinterpret_cast<const Zq*>(mat_b), nof_rows_b,
      nof_cols_b, config, reinterpret_cast<Zq*>(mat_out));
  }

  /**
   * @brief Scalar matrix multiplication (e.g., over Zq)
   */
  template <typename T>
  static eIcicleError cpu_matmul(
    const Device& device,
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out)
  {
    return cpu_matmul_internal<T, 1>(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
  }

  // Matrix-transpose implementation

  template <typename T>
  eIcicleError out_of_place_matrix_transpose(
    const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
  {
    const int nof_workers = get_nof_workers(config);
    tf::Taskflow taskflow;
    tf::Executor executor(nof_workers);

    const uint64_t elements_per_matrix = static_cast<uint64_t>(nof_rows) * nof_cols;

    for (uint32_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
      const T* mat_in_ptr = mat_in + batch_idx * elements_per_matrix;
      T* mat_out_ptr = mat_out + batch_idx * elements_per_matrix;

      for (uint32_t row = 0; row < nof_rows; ++row) {
        taskflow.emplace([=]() {
          for (uint32_t col = 0; col < nof_cols; ++col) {
            const uint64_t in_idx = row * nof_cols + col;
            const uint64_t out_idx = col * nof_rows + row;
            mat_out_ptr[out_idx] = mat_in_ptr[in_idx];
          }
        });
      }
    }

    executor.run(taskflow).wait();
    return eIcicleError::SUCCESS;
  }

  // Euclidean GCD
  static uint32_t gcd(uint32_t a, uint32_t b)
  {
    while (b != 0) {
      uint32_t temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  }

  // Mersenne modulo for rotation within total_bits
  uint64_t mersenne_mod(uint64_t shifted_idx, uint32_t total_bits)
  {
    uint64_t mod = (1ULL << total_bits) - 1;
    shifted_idx = (shifted_idx & mod) + (shifted_idx >> total_bits);
    while (shifted_idx >= mod) {
      shifted_idx = (shifted_idx & mod) + (shifted_idx >> total_bits);
    }
    return shifted_idx;
  }

  // Recursive generation of valid k-ary necklace start indices
  template <typename T>
  void gen_necklace(
    uint32_t t,
    uint32_t p,
    uint32_t k,
    uint32_t length,
    std::vector<uint32_t>& necklace,
    std::vector<uint64_t>& task_indices)
  {
    if (t > length) {
      if (
        length % p == 0 &&
        !std::all_of(necklace.begin() + 1, necklace.begin() + length + 1, [first = necklace[1]](uint32_t x) {
          return x == first;
        })) {
        uint64_t start_idx = 0;
        uint64_t multiplier = 1;
        for (int i = length; i >= 1; --i) {
          start_idx += necklace[i] * multiplier;
          multiplier *= k;
        }
        task_indices.push_back(start_idx);
      }
      return;
    }

    necklace[t] = necklace[t - p];
    gen_necklace<T>(t + 1, p, k, length, necklace, task_indices);

    for (uint32_t i = necklace[t - p] + 1; i < k; ++i) {
      necklace[t] = i;
      gen_necklace<T>(t + 1, t, k, length, necklace, task_indices);
    }
  }

  // Main element replacement logic based on necklace cycle walking
  template <typename T>
  void replace_elements(
    const T* input,
    uint64_t nof_operations,
    const std::vector<uint64_t>& start_indices_in_mat,
    uint64_t start_index,
    uint32_t log_nof_rows,
    uint32_t log_nof_cols,
    T* output)
  {
    const uint32_t total_bits = log_nof_rows + log_nof_cols;
    for (uint64_t i = 0; i < nof_operations; ++i) {
      const uint64_t start_idx = start_indices_in_mat[start_index + i];
      uint64_t idx = start_idx;
      T prev = input[idx];
      do {
        uint64_t shifted_idx = idx << log_nof_rows;
        uint64_t new_idx = mersenne_mod(shifted_idx, total_bits);
        T next = input[new_idx];
        output[new_idx] = prev;
        prev = next;
        idx = new_idx;
      } while (idx != start_idx);
    }
  }

  // Necklace-based transpose entry point
  template <typename T>
  eIcicleError matrix_transpose_necklaces(
    const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
  {
    if ((nof_rows & (nof_rows - 1)) != 0 || (nof_cols & (nof_cols - 1)) != 0) {
      return eIcicleError::INVALID_ARGUMENT; // must be power of 2
    }

    uint32_t log_nof_rows = static_cast<uint32_t>(std::floor(std::log2(nof_rows)));
    uint32_t log_nof_cols = static_cast<uint32_t>(std::floor(std::log2(nof_cols)));
    const uint32_t gcd_value = gcd(log_nof_rows, log_nof_cols);
    const uint32_t k = 1u << gcd_value;
    uint32_t length = (log_nof_cols + log_nof_rows) / gcd_value;

    std::vector<uint32_t> necklace(length + 1, 0);
    std::vector<uint64_t> start_indices_in_mat;
    gen_necklace<T>(1, 1, k, length, necklace, start_indices_in_mat);

    const int nof_workers = get_nof_workers(config);
    const uint64_t total_elements_one_mat = static_cast<uint64_t>(nof_rows) * nof_cols;
    // nof-tasks is a heuristic to balance the workload across workers
    const int nof_tasks = nof_workers * 2;
    const uint64_t max_nof_operations = (total_elements_one_mat + nof_tasks - 1) / nof_tasks;

    tf::Taskflow taskflow;
    tf::Executor executor(nof_workers);

    for (uint64_t i = 0; i < start_indices_in_mat.size(); i += max_nof_operations) {
      uint64_t nof_ops = std::min(max_nof_operations, start_indices_in_mat.size() - i);

      for (uint64_t batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
        const T* input_ptr = mat_in + batch_idx * total_elements_one_mat;
        T* output_ptr = mat_out + batch_idx * total_elements_one_mat;

        taskflow.emplace([=]() {
          replace_elements<T>(input_ptr, nof_ops, start_indices_in_mat, i, log_nof_rows, log_nof_cols, output_ptr);
        });
      }
    }

    executor.run(taskflow).wait();
    return eIcicleError::SUCCESS;
  }

  template <typename T>
  eIcicleError cpu_matrix_transpose(
    const Device& device, const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
  {
    if (!mat_in || !mat_out || nof_rows == 0 || nof_cols == 0) {
      ICICLE_LOG_ERROR << "Matrix-transpose: Invalid pointer or size";
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (config.columns_batch) {
      ICICLE_LOG_ERROR << "Matrix-transpose does not support columns_batch";
      return eIcicleError::INVALID_ARGUMENT;
    }

    // check if the number of rows and columns are powers of 2, if not use the basic transpose
    bool is_inplace = mat_in == mat_out;
    if (is_inplace) {
      bool is_power_of_2 = (nof_rows & (nof_rows - 1)) == 0 && (nof_cols & (nof_cols - 1)) == 0;
      if (is_power_of_2) {
        return (matrix_transpose_necklaces<T>(mat_in, nof_rows, nof_cols, config, mat_out));
      } else {
        // Copy the input matrix to a temporary vector and compute out-of-place transpose
        std::vector<T> mat_in_copy(nof_rows * nof_cols * config.batch_size);
        std::memcpy(mat_in_copy.data(), mat_in, nof_rows * nof_cols * config.batch_size * sizeof(T));
        return out_of_place_matrix_transpose(device, mat_in_copy.data(), nof_rows, nof_cols, config, mat_out);
      }
    }

    return out_of_place_matrix_transpose(device, mat_in, nof_rows, nof_cols, config, mat_out);
  }

} // namespace

// === Registration with runtime ===
REGISTER_MATMUL_BACKEND("CPU", cpu_matmul<field_config::scalar_t>);
REGISTER_MATRIX_TRANSPOSE_BACKEND("CPU", cpu_matrix_transpose<field_config::scalar_t>);
#ifdef EXT_FIELD
REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND("CPU", (cpu_matrix_transpose<field_config::extension_t>));
#endif
#ifdef RING
REGISTER_POLY_RING_MATMUL_BACKEND("CPU", (cpu_matmul_polynomial_ring<field_config::PolyRing>));
REGISTER_MATRIX_TRANSPOSE_RING_RNS_BACKEND("CPU", cpu_matrix_transpose<field_config::scalar_rns_t>);
REGISTER_MATRIX_TRANSPOSE_POLY_RING_BACKEND("CPU", cpu_matrix_transpose<field_config::PolyRing>);
#endif
