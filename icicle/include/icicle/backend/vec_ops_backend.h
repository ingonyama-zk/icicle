#pragma once

#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

  using vectorVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* scalar_a,
    const scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using vectorVectorOpImplInplaceA = std::function<eIcicleError(
    const Device& device, scalar_t* vec_a, const scalar_t* vec_b, uint64_t size, const VecOpsConfig& config)>;

  using scalarConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t size,
    bool is_to_montgomery,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using VectorReduceOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* scalar_a,
    const scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using scalarMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    scalar_t* out)>;

  using scalarBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  using scalarSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig& config,
    scalar_t* output)>;

  using scalarHighNonZeroIdxOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)>;

  using scalarPolyEvalImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* coeffs,
    uint64_t coeffs_size,
    const scalar_t* domain,
    uint64_t domain_size,
    const VecOpsConfig& config,
    scalar_t* evals /*OUT*/)>;

  using scalarPolyDivImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* numerator,
    uint64_t numerator_size,
    const scalar_t* denominator,
    uint64_t denominator_size,
    const VecOpsConfig& config,
    scalar_t* q_out /*OUT*/,
    uint64_t q_size,
    scalar_t* r_out /*OUT*/,
    uint64_t r_size)>;

  void register_vector_add(const std::string& deviceType, vectorVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add) = []() -> bool {                                                                  \
      register_vector_add(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_accumulate(const std::string& deviceType, vectorVectorOpImplInplaceA impl);

#define REGISTER_VECTOR_ACCUMULATE_BACKEND(DEVICE_TYPE, FUNC)                                                          \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_accumulate) = []() -> bool {                                                           \
      register_vector_accumulate(DEVICE_TYPE, FUNC);                                                                   \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sub(const std::string& deviceType, vectorVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub) = []() -> bool {                                                                  \
      register_vector_sub(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_mul(const std::string& deviceType, vectorVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul) = []() -> bool {                                                                  \
      register_vector_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_div(const std::string& deviceType, vectorVectorOpImpl impl);

#define REGISTER_VECTOR_DIV_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_div) = []() -> bool {                                                                  \
      register_vector_div(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_convert_montgomery(const std::string& deviceType, scalarConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sum(const std::string& deviceType, VectorReduceOpImpl impl);

#define REGISTER_VECTOR_SUM_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sum) = []() -> bool {                                                                  \
      register_vector_sum(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_product(const std::string& deviceType, VectorReduceOpImpl impl);

#define REGISTER_VECTOR_PRODUCT_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_product) = []() -> bool {                                                              \
      register_vector_product(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_mul_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_MUL_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_mul_vec) = []() -> bool {                                                           \
      register_scalar_mul_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_add_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_ADD_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_add_vec) = []() -> bool {                                                           \
      register_scalar_add_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_sub_vec(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_SUB_VEC_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_sub_vec) = []() -> bool {                                                           \
      register_scalar_sub_vec(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_matrix_transpose(const std::string& deviceType, scalarMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose) = []() -> bool {                                                         \
      register_matrix_transpose(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_bit_reverse(const std::string& deviceType, scalarBitReverseOpImpl);

#define REGISTER_BIT_REVERSE_BACKEND(DEVICE_TYPE, FUNC)                                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_bit_reverse) = []() -> bool {                                                       \
      register_scalar_bit_reverse(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_slice(const std::string& deviceType, scalarSliceOpImpl);

#define REGISTER_SLICE_BACKEND(DEVICE_TYPE, FUNC)                                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                             \
      register_slice(DEVICE_TYPE, FUNC);                                                                               \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_highest_non_zero_idx(const std::string& deviceType, scalarHighNonZeroIdxOpImpl);

#define REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_highest_non_zero_idx) = []() -> bool {                                              \
      register_highest_non_zero_idx(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_poly_eval(const std::string& deviceType, scalarPolyEvalImpl);

#define REGISTER_POLYNOMIAL_EVAL(DEVICE_TYPE, FUNC)                                                                    \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poly_eval) = []() -> bool {                                                                \
      register_poly_eval(DEVICE_TYPE, FUNC);                                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_poly_division(const std::string& deviceType, scalarPolyDivImpl);

#define REGISTER_POLYNOMIAL_DIVISION(DEVICE_TYPE, FUNC)                                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poly_division) = []() -> bool {                                                            \
      register_poly_division(DEVICE_TYPE, FUNC);                                                                       \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef EXT_FIELD
  using extFieldVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* vec_a,
    const extension_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    extension_t* output)>;

  using extFieldVectorOpImplInplaceA = std::function<eIcicleError(
    const Device& device, extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig& config)>;

  void register_extension_vector_add(const std::string& deviceType, extFieldVectorOpImpl impl);

  #define REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_add_ext_field) = []() -> bool {                                                      \
        register_extension_vector_add(DEVICE_TYPE, FUNC);                                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_vector_accumulate(const std::string& deviceType, extFieldVectorOpImplInplaceA impl);

  #define REGISTER_VECTOR_ACCUMULATE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                              \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_accumulate_ext_field) = []() -> bool {                                               \
        register_extension_vector_accumulate(DEVICE_TYPE, FUNC);                                                       \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_vector_sub(const std::string& deviceType, extFieldVectorOpImpl impl);
  #define REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_sub_ext_field) = []() -> bool {                                                      \
        register_extension_vector_sub(DEVICE_TYPE, FUNC);                                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_vector_mul(const std::string& deviceType, extFieldVectorOpImpl impl);

  #define REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_mul_ext_field) = []() -> bool {                                                      \
        register_extension_vector_mul(DEVICE_TYPE, FUNC);                                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using extFieldConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    uint64_t size,
    bool is_to_montgomery,
    const VecOpsConfig& config,
    extension_t* output)>;

  void register_extension_scalar_convert_montgomery(const std::string& deviceType, extFieldConvertMontgomeryImpl);

  #define REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                             \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_convert_mont_ext_field) = []() -> bool {                                          \
        register_extension_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

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

  using extFieldBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const extension_t* input, uint64_t size, const VecOpsConfig& config, extension_t* output)>;

  void register_extension_bit_reverse(const std::string& deviceType, extFieldBitReverseOpImpl);

  #define REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                    \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                    \
        register_extension_bit_reverse(DEVICE_TYPE, FUNC);                                                             \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using extFieldSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size,
    const VecOpsConfig& config,
    extension_t* output)>;

  void register_extension_slice(const std::string& deviceType, extFieldSliceOpImpl);

  #define REGISTER_SLICE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                          \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                           \
        register_extension_slice(DEVICE_TYPE, FUNC);                                                                   \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // EXT_FIELD

} // namespace icicle