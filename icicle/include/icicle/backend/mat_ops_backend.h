#pragma once

#include "icicle/vec_ops.h"
#include "icicle/mat_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration for matrix operations ***************************/

  using scalarBinaryMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    scalar_t* mat_out)>;

  void register_matmul(const std::string& deviceType, scalarBinaryMatrixOpImpl impl);

#define REGISTER_MATMUL_BACKEND(DEVICE_TYPE, FUNC)                                                                     \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matmul) = []() -> bool {                                                                   \
      register_matmul(DEVICE_TYPE, FUNC);                                                                              \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef RING
  using polyRingBinaryMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const PolyRing* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const PolyRing* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    PolyRing* mat_out)>;

  void register_poly_ring_matmul(const std::string& deviceType, polyRingBinaryMatrixOpImpl impl);
  #define REGISTER_POLY_RING_MATMUL_BACKEND(DEVICE_TYPE, FUNC)                                                         \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_poly_ring_matmul) = []() -> bool {                                                       \
        register_poly_ring_matmul(DEVICE_TYPE, FUNC);                                                                  \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // RING

} // namespace icicle