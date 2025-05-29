#pragma once

#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

  using scalarUnaryMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    scalar_t* out)>;

  using scalarBinaryMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* in_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    scalar_t* out)>;

  using tqBinaryMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    uint32_t d,
    const scalar_t* in_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const scalar_t* in_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const VecOpsConfig& config,
    scalar_t* out)>;

  void register_matrix_transpose(const std::string& deviceType, scalarUnaryMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose) = []() -> bool {                                                         \
      register_matrix_transpose(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_matrix_mult(const std::string& deviceType, scalarBinaryMatrixOpImpl impl);

#define REGISTER_MATRIX_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_mul) = []() -> bool {                                                               \
      register_matrix_mult(DEVICE_TYPE, FUNC);                                                                         \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_tq_matrix_mult(const std::string& deviceType, tqBinaryMatrixOpImpl impl);

#define REGISTER_TQ_MATRIX_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                              \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_tq_matrix_mul) = []() -> bool {                                                            \
      register_tq_matrix_mult(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef EXT_FIELD
  using extFieldMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    extension_t* out)>;

  void register_extension_matrix_transpose(const std::string& deviceType, extFieldMatrixOpImpl impl);

  #define REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                               \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_matrix_transpose_ext_field) = []() -> bool {                                             \
        register_extension_matrix_transpose(DEVICE_TYPE, FUNC);                                                        \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // EXT_FIELD

#ifdef RING // for RNS type
  using ringRnsMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_rns_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    scalar_rns_t* out)>;

  void register_ring_rns_matrix_transpose(const std::string& deviceType, ringRnsMatrixOpImpl impl);

  #define REGISTER_MATRIX_TRANSPOSE_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_matrix_transpose_ring_rns) = []() -> bool {                                              \
        register_ring_rns_matrix_transpose(DEVICE_TYPE, FUNC);                                                         \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // RING

} // namespace icicle