#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"
#include <cstdint>
#include <vector>
#include <cstring>

#include "taskflow/taskflow.hpp"
#include "icicle/program/program.h"
#include "cpu_program_executor.h"

// Extract number of threads to run from configuration
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

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
  template <typename Zq, uint32_t degree>
  static eIcicleError cpu_matrix_mult_internal(
    const Zq* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const Zq* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    Zq* mat_out)
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
          Zq acc[degree];
          std::memset(acc, 0, sizeof(acc));

          // Compute dot product of row from A and column from B
          for (uint32_t k = 0; k < nof_cols_a; ++k) {
            const Zq* a = mat_a + (row * nof_cols_a + k) * degree;
            const Zq* b = mat_b + (k * nof_cols_b + col) * degree;
            for (uint32_t d = 0; d < degree; ++d) {
              acc[d] = acc[d] + a[d] * b[d];
            }
          }

          Zq* out = mat_out + (row * nof_cols_b + col) * degree;
          std::memcpy(out, acc, sizeof(Zq) * degree);
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
  static eIcicleError cpu_matrix_mult_polynomial_ring(
    const Device& device,
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    T* mat_out)
  {
    using Zq = typename T::Base;
    constexpr uint32_t d = T::d;

    return cpu_matrix_mult_internal<Zq, d>(
      reinterpret_cast<const Zq*>(mat_a), nof_rows_a, nof_cols_a, reinterpret_cast<const Zq*>(mat_b), nof_rows_b,
      nof_cols_b, config, reinterpret_cast<Zq*>(mat_out));
  }

  /**
   * @brief Scalar matrix multiplication (e.g., over Zq)
   */
  template <typename T>
  static eIcicleError cpu_matrix_mult(
    const Device& device,
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    T* mat_out)
  {
    return cpu_matrix_mult_internal<T, 1>(
      mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
  }

} // namespace

// === Registration with runtime ===
REGISTER_MATRIX_MULT_BACKEND("CPU", cpu_matrix_mult<scalar_t>);
#ifdef RING
REGISTER_POLY_RING_MATRIX_MULT_BACKEND("CPU", (cpu_matrix_mult_polynomial_ring<PolyRing>));
#endif