#pragma once

#include "icicle/vec_ops.h"
#include "icicle/fields/field_config.h"
#include "icicle/norm.h"
using namespace field_config;

namespace icicle {
  /*************************** Backend registration ***************************/

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

  using programExecutionImpl = std::function<eIcicleError(
    const Device& device,
    std::vector<scalar_t*>& data,
    const Program<scalar_t>& program,
    uint64_t size,
    const VecOpsConfig& config)>;

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

  void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);

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

  void register_vector_inv(const std::string& deviceType, VectorReduceOpImpl impl);

#define REGISTER_VECTOR_INV_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_inv) = []() -> bool {                                                                  \
      register_vector_inv(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub) = []() -> bool {                                                                  \
      register_vector_sub(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul) = []() -> bool {                                                                  \
      register_vector_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_div(const std::string& deviceType, scalarVectorOpImpl impl);

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

  void register_execute_program(const std::string& deviceType, programExecutionImpl);

#define REGISTER_EXECUTE_PROGRAM_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_program_execution) = []() -> bool {                                                        \
      register_execute_program(DEVICE_TYPE, FUNC);                                                                     \
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

  using mixedVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* scalar_a,
    const scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    extension_t* output)>;

  using extFieldVectorOpImplInplaceA = std::function<eIcicleError(
    const Device& device, extension_t* vec_a, const extension_t* vec_b, uint64_t size, const VecOpsConfig& config)>;

  using extFieldVectorReduceOpImpl = std::function<eIcicleError(
    const Device& device, const extension_t* vec_a, uint64_t size, const VecOpsConfig& config, extension_t* output)>;

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

  void register_extension_vector_inv(const std::string& deviceType, extFieldVectorReduceOpImpl impl);

  #define REGISTER_VECTOR_INV_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_inv_ext_field) = []() -> bool {                                                      \
        register_extension_vector_inv(DEVICE_TYPE, FUNC);                                                              \
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

  void register_extension_vector_mixed_mul(const std::string& deviceType, mixedVectorOpImpl impl);

  #define REGISTER_VECTOR_MIXED_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                         \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_mixed_mul) = []() -> bool {                                                          \
        register_extension_vector_mixed_mul(DEVICE_TYPE, FUNC);                                                        \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_vector_div(const std::string& deviceType, extFieldVectorOpImpl impl);

  #define REGISTER_VECTOR_DIV_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_div_ext_field) = []() -> bool {                                                      \
        register_extension_vector_div(DEVICE_TYPE, FUNC);                                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_scalar_mul_vec(const std::string& deviceType, extFieldVectorOpImpl impl);

  #define REGISTER_SCALAR_MUL_VEC_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_mul_vec_ext_field) = []() -> bool {                                               \
        register_extension_scalar_mul_vec(DEVICE_TYPE, FUNC);                                                          \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_scalar_add_vec(const std::string& deviceType, extFieldVectorOpImpl impl);

  #define REGISTER_SCALAR_ADD_VEC_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_add_vec_ext_field) = []() -> bool {                                               \
        register_extension_scalar_add_vec(DEVICE_TYPE, FUNC);                                                          \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_scalar_sub_vec(const std::string& deviceType, extFieldVectorOpImpl impl);

  #define REGISTER_SCALAR_SUB_VEC_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_sub_vec_ext_field) = []() -> bool {                                               \
        register_extension_scalar_sub_vec(DEVICE_TYPE, FUNC);                                                          \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_vector_sum(const std::string& deviceType, extFieldVectorReduceOpImpl impl);

  #define REGISTER_VECTOR_SUM_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_sum_ext_field) = []() -> bool {                                                      \
        register_extension_vector_sum(DEVICE_TYPE, FUNC);                                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_extension_vector_product(const std::string& deviceType, extFieldVectorReduceOpImpl impl);

  #define REGISTER_VECTOR_PRODUCT_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_product_ext_field) = []() -> bool {                                                  \
        register_extension_vector_product(DEVICE_TYPE, FUNC);                                                          \
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
    uint64_t size_in,
    uint64_t size_out,
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

  using extProgramExecutionImpl = std::function<eIcicleError(
    const Device& device,
    std::vector<extension_t*>& data,
    const Program<extension_t>& program,
    uint64_t size,
    const VecOpsConfig& config)>;

  void register_extension_execute_program(const std::string& deviceType, extProgramExecutionImpl);

  #define REGISTER_EXECUTE_PROGRAM_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_program_execution) = []() -> bool {                                                      \
        register_extension_execute_program(DEVICE_TYPE, FUNC);                                                         \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // EXT_FIELD

#ifdef RING
  // for Zq type

  // This should be the same for all the devices to get a deterministic result
  const size_t RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS = 256;

  using ringZqRandomSamplingImpl = std::function<eIcicleError(
    const Device& device,
    size_t size,
    bool fast_mode,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& cfg,
    field_t* output)>;
  void register_ring_zq_random_sampling(const std::string& deviceType, ringZqRandomSamplingImpl);

  #define REGISTER_RING_ZQ_RANDOM_SAMPLING_BACKEND(DEVICE_TYPE, FUNC)                                                  \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_ring_zq_random_sampling) = []() -> bool {                                                \
        register_ring_zq_random_sampling(DEVICE_TYPE, FUNC);                                                           \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  // for Rq type

  using challengeSpacePolynomialsSamplingImpl = std::function<eIcicleError(
    const Device& device,
    const std::byte* seed,
    size_t seed_len,
    size_t size,
    uint32_t ones,
    uint32_t twos,
    uint64_t norm,
    const VecOpsConfig& cfg,
    Rq* output)>;
  void
  register_challenge_space_polynomials_sampling(const std::string& deviceType, challengeSpacePolynomialsSamplingImpl);

  #define REGISTER_CHALLENGE_SPACE_POLYNOMIALS_SAMPLING_BACKEND(DEVICE_TYPE, FUNC)                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_challenge_space_polynomials_sampling) = []() -> bool {                                   \
        register_challenge_space_polynomials_sampling(DEVICE_TYPE, FUNC);                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  // for RNS type
  using ringRnsVectorReduceOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_rns_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)>;
  using ringRnsVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_rns_t* vec_a,
    const scalar_rns_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)>;

  using mixedVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_rns_t* scalar_a,
    const scalar_t* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_rns_t* output)>;

  using ringRnsVectorOpImplInplaceA = std::function<eIcicleError(
    const Device& device, scalar_rns_t* vec_a, const scalar_rns_t* vec_b, uint64_t size, const VecOpsConfig& config)>;

  using ringRnsVectorReduceOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_rns_t* vec_a, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)>;

  void register_ring_rns_vector_add(const std::string& deviceType, ringRnsVectorOpImpl impl);

  #define REGISTER_VECTOR_ADD_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                      \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_add_ring_rns) = []() -> bool {                                                       \
        register_ring_rns_vector_add(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_accumulate(const std::string& deviceType, ringRnsVectorOpImplInplaceA impl);

  #define REGISTER_VECTOR_ACCUMULATE_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                               \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_accumulate_ring_rns) = []() -> bool {                                                \
        register_ring_rns_vector_accumulate(DEVICE_TYPE, FUNC);                                                        \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_sub(const std::string& deviceType, ringRnsVectorOpImpl impl);
  #define REGISTER_VECTOR_SUB_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                      \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_sub_ring_rns) = []() -> bool {                                                       \
        register_ring_rns_vector_sub(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_mul(const std::string& deviceType, ringRnsVectorOpImpl impl);

  #define REGISTER_VECTOR_MUL_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                      \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_mul_ring_rns) = []() -> bool {                                                       \
        register_ring_rns_vector_mul(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_mixed_mul(const std::string& deviceType, mixedVectorOpImpl impl);

  #define REGISTER_VECTOR_MIXED_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                         \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_mixed_mul) = []() -> bool {                                                          \
        register_ring_rns_vector_mixed_mul(DEVICE_TYPE, FUNC);                                                         \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_div(const std::string& deviceType, ringRnsVectorOpImpl impl);

  #define REGISTER_VECTOR_DIV_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                      \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_div_ring_rns) = []() -> bool {                                                       \
        register_ring_rns_vector_div(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_inv(const std::string& deviceType, ringRnsVectorReduceOpImpl impl);

  #define REGISTER_VECTOR_INV_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                      \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_inv_ring_rns) = []() -> bool {                                                       \
        register_ring_rns_vector_inv(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_scalar_mul_vec(const std::string& deviceType, ringRnsVectorOpImpl impl);

  #define REGISTER_SCALAR_MUL_VEC_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                  \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_mul_vec_ring_rns) = []() -> bool {                                                \
        register_ring_rns_scalar_mul_vec(DEVICE_TYPE, FUNC);                                                           \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_scalar_add_vec(const std::string& deviceType, ringRnsVectorOpImpl impl);

  #define REGISTER_SCALAR_ADD_VEC_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                  \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_add_vec_ring_rns) = []() -> bool {                                                \
        register_ring_rns_scalar_add_vec(DEVICE_TYPE, FUNC);                                                           \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_scalar_sub_vec(const std::string& deviceType, ringRnsVectorOpImpl impl);

  #define REGISTER_SCALAR_SUB_VEC_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                  \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_sub_vec_ring_rns) = []() -> bool {                                                \
        register_ring_rns_scalar_sub_vec(DEVICE_TYPE, FUNC);                                                           \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_sum(const std::string& deviceType, ringRnsVectorReduceOpImpl impl);

  #define REGISTER_VECTOR_SUM_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                      \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_sum_ring_rns) = []() -> bool {                                                       \
        register_ring_rns_vector_sum(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  void register_ring_rns_vector_product(const std::string& deviceType, ringRnsVectorReduceOpImpl impl);

  #define REGISTER_VECTOR_PRODUCT_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                  \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_vec_product_ring_rns) = []() -> bool {                                                   \
        register_ring_rns_vector_product(DEVICE_TYPE, FUNC);                                                           \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using ringRnsConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_rns_t* input,
    uint64_t size,
    bool is_to_montgomery,
    const VecOpsConfig& config,
    scalar_rns_t* output)>;

  void register_ring_rns_scalar_convert_montgomery(const std::string& deviceType, ringRnsConvertMontgomeryImpl);

  #define REGISTER_CONVERT_MONTGOMERY_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                              \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_convert_mont_ring_rns) = []() -> bool {                                           \
        register_ring_rns_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

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

  using ringPolyRingMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const PolyRing* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    PolyRing* out)>;

  void register_poly_ring_matrix_transpose(const std::string& deviceType, ringPolyRingMatrixOpImpl impl);

  #define REGISTER_MATRIX_TRANSPOSE_POLY_RING_BACKEND(DEVICE_TYPE, FUNC)                                               \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_matrix_transpose_poly_ring) = []() -> bool {                                             \
        register_poly_ring_matrix_transpose(DEVICE_TYPE, FUNC);                                                        \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using ringRnsBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_rns_t* input, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)>;

  void register_ring_rns_bit_reverse(const std::string& deviceType, ringRnsBitReverseOpImpl);

  #define REGISTER_BIT_REVERSE_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                     \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                    \
        register_ring_rns_bit_reverse(DEVICE_TYPE, FUNC);                                                              \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using ringRnsSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_rns_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig& config,
    scalar_rns_t* output)>;

  void register_ring_rns_slice(const std::string& deviceType, ringRnsSliceOpImpl);

  #define REGISTER_SLICE_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                           \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                           \
        register_ring_rns_slice(DEVICE_TYPE, FUNC);                                                                    \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  // RNS <--> direct conversion
  using ringConvertToRnsImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_rns_t* output)>;

  void register_convert_to_rns(const std::string& deviceType, ringConvertToRnsImpl);
  #define REGISTER_CONVERT_TO_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                           \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_convert_to_rns) = []() -> bool {                                                         \
        register_convert_to_rns(DEVICE_TYPE, FUNC);                                                                    \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using ringConvertFromRnsImpl = std::function<eIcicleError(
    const Device& device, const scalar_rns_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  void register_convert_from_rns(const std::string& deviceType, ringConvertFromRnsImpl);
  #define REGISTER_CONVERT_FROM_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                         \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_convert_from_rns) = []() -> bool {                                                       \
        register_convert_from_rns(DEVICE_TYPE, FUNC);                                                                  \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using rnsProgramExecutionImpl = std::function<eIcicleError(
    const Device& device,
    std::vector<scalar_rns_t*>& data,
    const Program<scalar_rns_t>& program,
    uint64_t size,
    const VecOpsConfig& config)>;

  void register_rns_execute_program(const std::string& deviceType, rnsProgramExecutionImpl);

  #define REGISTER_EXECUTE_PROGRAM_RING_RNS_BACKEND(DEVICE_TYPE, FUNC)                                                 \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_program_execution_rns) = []() -> bool {                                                  \
        register_rns_execute_program(DEVICE_TYPE, FUNC);                                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using balancedDecompositionImpl = std::function<eIcicleError(
    const Device& device,
    const field_t* input,
    size_t input_size,
    uint32_t base,
    const VecOpsConfig& config,
    field_t* output,
    size_t output_size)>;

  void register_decompose_balanced_digits(const std::string& deviceType, balancedDecompositionImpl impl);
  void register_recompose_from_balanced_digits(const std::string& deviceType, balancedDecompositionImpl impl);

  #define REGISTER_BALANCED_DECOMPOSITION_BACKEND(DEVICE_TYPE, DECOMPOSE, RECOMPOSE)                                   \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_balanced_recomposition) = []() -> bool {                                                 \
        register_decompose_balanced_digits(DEVICE_TYPE, DECOMPOSE);                                                    \
        register_recompose_from_balanced_digits(DEVICE_TYPE, RECOMPOSE);                                               \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using JLProjectionImpl = std::function<eIcicleError(
    const Device& device,
    const field_t* input,
    size_t input_size,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& cfg,
    field_t* output,
    size_t output_size)>;

  using JLProjectionGetRowsImpl = std::function<eIcicleError(
    const Device& device,
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    bool negacyclic_conjugate,
    size_t polyring_size_for_conjugate,
    const VecOpsConfig& cfg,
    field_t* output)>;

  void register_jl_projection(const std::string& deviceType, JLProjectionImpl impl);
  void register_jl_projection_get_rows(const std::string& deviceType, JLProjectionGetRowsImpl impl);
  #define REGISTER_JL_PROJECTION_BACKEND(DEVICE_TYPE, PROJECTION, GET_ROWS)                                            \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_jl_projection) = []() -> bool {                                                          \
        register_jl_projection(DEVICE_TYPE, PROJECTION);                                                               \
        register_jl_projection_get_rows(DEVICE_TYPE, GET_ROWS);                                                        \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  // Norm checking implementations
  using normCheckImpl = std::function<eIcicleError(
    const Device& device,
    const field_t* input,
    size_t size,
    eNormType norm,
    uint64_t norm_bound,
    const VecOpsConfig& config,
    bool* output)>;

  using normCheckRelativeImpl = std::function<eIcicleError(
    const Device& device,
    const field_t* input_a,
    const field_t* input_b,
    size_t size,
    eNormType norm,
    uint64_t scale,
    const VecOpsConfig& config,
    bool* output)>;

  void register_check_norm_bound(const std::string& deviceType, normCheckImpl impl);
  void register_check_norm_relative(const std::string& deviceType, normCheckRelativeImpl impl);

  #define REGISTER_NORM_CHECK_BACKEND(DEVICE_TYPE, CHECK_BOUND, CHECK_RELATIVE)                                        \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_norm_check) = []() -> bool {                                                             \
        register_check_norm_bound(DEVICE_TYPE, CHECK_BOUND);                                                           \
        register_check_norm_relative(DEVICE_TYPE, CHECK_RELATIVE);                                                     \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // RING
} // namespace icicle