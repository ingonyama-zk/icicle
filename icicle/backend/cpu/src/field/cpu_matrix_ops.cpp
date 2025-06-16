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

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

namespace {
  using namespace field_config;
  using namespace icicle;

  // Matrix multiplication where each element is logically a T[degree]
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
    if (!mat_a || !mat_b || !mat_out || nof_rows_a == 0 || nof_cols_a == 0 || nof_rows_b == 0 || nof_cols_b == 0)
      return eIcicleError::INVALID_ARGUMENT;

    if (config.batch_size != 1) {
      ICICLE_LOG_ERROR << "Matmul does not support batch";
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (nof_cols_a != nof_rows_b) return eIcicleError::INVALID_ARGUMENT;

    const int nof_workers = get_nof_workers(config);

    tf::Taskflow taskflow;
    tf::Executor executor(nof_workers);

    for (uint32_t row = 0; row < nof_rows_a; ++row) {
      taskflow.emplace([=]() {
        for (uint32_t col = 0; col < nof_cols_b; ++col) {
          Zq acc[degree];
          std::memset(acc, 0, sizeof(acc));

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

  // Specialization for PolyRing<T, d>
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

  // Scalar matmul for `Zq` or any element with degree 1
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

// === Registration ===
REGISTER_MATRIX_MULT_BACKEND("CPU", cpu_matrix_mult<scalar_t>);
#ifdef RING
REGISTER_POLY_RING_MATRIX_MULT_BACKEND("CPU", (cpu_matrix_mult_polynomial_ring<PolyRing>));
#endif