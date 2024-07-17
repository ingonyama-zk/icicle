#pragma once

#include <functional>

#include "errors.h"
#include "runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

using namespace field_config;

namespace icicle {

  /*************************** Frontend APIs ***************************/
  /**
   * @brief Configuration for vector operations.
   * @note APIs with a single input, ignore input b.
   */
  struct VecOpsConfig {
    icicleStreamHandle stream; /**< Stream for asynchronous execution. */
    bool is_a_on_device;       /**< True if `a` is on the device, false if it is not. Default value: false. */
    bool is_b_on_device;       /**< True if `b` is on the device, false if it is not. Default value: false. OPTIONAL. */
    bool is_result_on_device; /**< If true, the output is preserved on the device, otherwise on the host. Default value:
                                 false. */
    bool is_async;            /**< Whether to run the vector operations asynchronously.
                                   If set to `true`, the function will be non-blocking and synchronization
                                   must be explicitly managed using `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                   If set to `false`, the function will block the current CPU thread. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extension. */
  };

  /**
   * @brief Returns the default value of VecOpsConfig.
   *
   * @return Default value of VecOpsConfig.
   */
  static VecOpsConfig default_vec_ops_config()
  {
    VecOpsConfig config = {
      nullptr, // stream
      false,   // is_a_on_device
      false,   // is_b_on_device
      false,   // is_result_on_device
      false,   // is_async
    };
    return config;
  }

  // Element-wise vector operations

  /**
   * @brief Adds two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Input vector `a`.
   * @param vec_b Input vector `b`.
   * @param size Number of elements in the vectors.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_add(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Subtracts vector `b` from vector `a` element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Input vector `a`.
   * @param vec_b Input vector `b`.
   * @param size Number of elements in the vectors.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_sub(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Input vector `a`.
   * @param vec_b Input vector `b`.
   * @param size Number of elements in the vectors.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_mul(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Divides vector `a` by vector `b` element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Input vector `a`.
   * @param vec_b Input vector `b`.
   * @param size Number of elements in the vectors.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_div(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Converts elements to and from Montgomery form.
   *
   * @tparam T Type of the elements.
   * @param input Input vector.
   * @param size Number of elements in the input vector.
   * @param is_into True to convert into Montgomery form, false to convert out of Montgomery form.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError convert_montgomery(const T* input, uint64_t size, bool is_into, const VecOpsConfig& config, T* output);

  // Scalar-Vector operations

  /**
   * @brief Adds a scalar to each element of a vector.
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Input scalar.
   * @param vec_b Input vector.
   * @param size Number of elements in the vector.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError scalar_add_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Subtracts each element of a vector from a scalar, elementwise (res[i]=scalar-vec[i]).
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Input scalar.
   * @param vec_b Input vector.
   * @param size Number of elements in the vector.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError scalar_sub_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies each element of a vector by a scalar.
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Input scalar.
   * @param vec_b Input vector.
   * @param size Number of elements in the vector.
   * @param config Configuration for the operation.
   * @param output Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError scalar_mul_vec(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  // Matrix operations

  /**
   * @brief Transposes a matrix.
   *
   * @tparam T Type of the elements in the matrix.
   * @param mat_in Input matrix.
   * @param nof_rows Number of rows in the input matrix.
   * @param nof_cols Number of columns in the input matrix.
   * @param config Configuration for the operation.
   * @param mat_out Output matrix to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError
  matrix_transpose(const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out);

  // Miscellaneous operations

  /**
   * @brief Reorders the vector elements based on bit-reverse. That is out[i]=in[bitrev[i]].
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_in Input vector.
   * @param size Number of elements in the input vector.
   * @param config Configuration for the operation.
   * @param vec_out Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError bit_reverse(const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out);

  /**
   * @brief Extracts a slice from a vector.
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_in Input vector.
   * @param offset Offset from which to start the slice.
   * @param stride Stride between elements in the slice.
   * @param size Number of elements in the slice.
   * @param config Configuration for the operation.
   * @param vec_out Output vector to store the result.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError
  slice(const T* vec_in, uint64_t offset, uint64_t stride, uint64_t size, const VecOpsConfig& config, T* vec_out);

  /**
   * @brief Finds the highest non-zero index in a vector.
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_in Input vector.
   * @param size Number of elements in the input vector.
   * @param config Configuration for the operation.
   * @param out_idx Output index of the highest non-zero element.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError highest_non_zero_idx(const T* vec_in, uint64_t size, const VecOpsConfig& config, int64_t* out_idx);

  /**
   * @brief Evaluates a polynomial at given domain points.
   *
   * @tparam T Type of the elements in the polynomial and domain.
   * @param coeffs Pointer to the array of coefficients of the polynomial.
   * @param coeffs_size Number of coefficients in the polynomial.
   * @param domain Pointer to the array of points at which to evaluate the polynomial.
   * @param domain_size Number of domain points.
   * @param config Configuration for the operation.
   * @param evals Pointer to the array where the evaluated results will be stored. This is an output parameter.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError polynomial_eval(
    const T* coeffs,
    uint64_t coeffs_size,
    const T* domain,
    uint64_t domain_size,
    const VecOpsConfig& config,
    T* evals /*OUT*/);

  /**
   * @brief Divides two polynomials.
   *
   * @tparam T Type of the elements in the polynomials.
   * @param numerator Pointer to the array of coefficients of the numerator polynomial.
   * @param numerator_deg Degree of the numerator polynomial.
   * @param denominator Pointer to the array of coefficients of the denominator polynomial.
   * @param denominator_deg Degree of the denominator polynomial.
   * @param config Configuration for the operation.
   * @param q_out Pointer to the array where the quotient will be stored. This is an output parameter.
   * @param q_size Size of the quotient array.
   * @param r_out Pointer to the array where the remainder will be stored. This is an output parameter.
   * @param r_size Size of the remainder array.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError polynomial_division(
    const T* numerator,
    int64_t numerator_deg,
    const T* denumerator,
    int64_t denumerator_deg,
    const VecOpsConfig& config,
    T* q_out /*OUT*/,
    uint64_t q_size,
    T* r_out /*OUT*/,
    uint64_t r_size);

  /*************************** Backend registration ***************************/

  using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* vec_a,
    const scalar_t* vec_b,
    uint64_t n,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add) = []() -> bool {                                                                  \
      register_vector_add(DEVICE_TYPE, FUNC);                                                                          \
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

  using scalarConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t size,
    bool is_into,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_scalar_convert_montgomery(const std::string& deviceType, scalarConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    scalar_t* out)>;

  void register_matrix_transpose(const std::string& deviceType, scalarMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose) = []() -> bool {                                                         \
      register_matrix_transpose(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  void register_scalar_bit_reverse(const std::string& deviceType, scalarBitReverseOpImpl);

#define REGISTER_BIT_REVERSE_BACKEND(DEVICE_TYPE, FUNC)                                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_bit_reverse) = []() -> bool {                                                       \
      register_scalar_bit_reverse(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_slice(const std::string& deviceType, scalarSliceOpImpl);

#define REGISTER_SLICE_BACKEND(DEVICE_TYPE, FUNC)                                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                             \
      register_slice(DEVICE_TYPE, FUNC);                                                                               \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarHighNonZeroIdxOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, int64_t* out_idx /*OUT*/)>;

  void register_highest_non_zero_idx(const std::string& deviceType, scalarHighNonZeroIdxOpImpl);

#define REGISTER_HIGHEST_NON_ZERO_IDX_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_highest_non_zero_idx) = []() -> bool {                                              \
      register_highest_non_zero_idx(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  template <typename T>
  eIcicleError polynomial_eval(
    const T* coeffs,
    uint64_t coeffs_size,
    const T* domain,
    uint64_t domain_size,
    const VecOpsConfig& config,
    T* evals /*OUT*/);

  using scalarPolyEvalImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* coeffs,
    uint64_t coeffs_size,
    const scalar_t* domain,
    uint64_t domain_size,
    const VecOpsConfig& config,
    scalar_t* evals /*OUT*/)>;

  void register_poly_eval(const std::string& deviceType, scalarPolyEvalImpl);

#define REGISTER_POLYNOMIAL_EVAL(DEVICE_TYPE, FUNC)                                                                    \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poly_eval) = []() -> bool {                                                                \
      register_poly_eval(DEVICE_TYPE, FUNC);                                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarPolyDivImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* numerator,
    int64_t numerator_deg,
    const scalar_t* denumerator,
    int64_t denumerator_deg,
    const VecOpsConfig& config,
    scalar_t* q_out /*OUT*/,
    uint64_t q_size,
    scalar_t* r_out /*OUT*/,
    uint64_t r_size)>;

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
    uint64_t n,
    const VecOpsConfig& config,
    extension_t* output)>;

  void register_extension_vector_add(const std::string& deviceType, extFieldVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add_ext_field) = []() -> bool {                                                        \
      register_extension_vector_add(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_extension_vector_sub(const std::string& deviceType, extFieldVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub_ext_field) = []() -> bool {                                                        \
      register_extension_vector_sub(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_extension_vector_mul(const std::string& deviceType, extFieldVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul_ext_field) = []() -> bool {                                                        \
      register_extension_vector_mul(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    uint64_t size,
    bool is_into,
    const VecOpsConfig& config,
    extension_t* output)>;

  extern "C" void
  register_extension_scalar_convert_montgomery(const std::string& deviceType, extFieldConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont_ext_field) = []() -> bool {                                            \
      register_extension_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    extension_t* out)>;

  void register_extension_matrix_transpose(const std::string& deviceType, extFieldMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose_ext_field) = []() -> bool {                                               \
      register_extension_matrix_transpose(DEVICE_TYPE, FUNC);                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const extension_t* input, uint64_t size, const VecOpsConfig& config, extension_t* output)>;

  void register_extension_bit_reverse(const std::string& deviceType, extFieldBitReverseOpImpl);

#define REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_extension_bit_reverse(DEVICE_TYPE, FUNC);                                                               \
      return true;                                                                                                     \
    }();                                                                                                               \
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

#define REGISTER_SLICE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                             \
      register_extension_slice(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // EXT_FIELD

} // namespace icicle