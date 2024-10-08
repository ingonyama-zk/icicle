#pragma once

#include <functional>

#include "errors.h"
#include "runtime.h"

#include "icicle/fields/field.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

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
                                                       must be explicitly managed using `cudaStreamSynchronize` or
                                 `cudaDeviceSynchronize`.            If set to `false`, the function will block the current CPU thread. */
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
   * @brief Accumulates the elements of two vectors element-wise and stores the result in the first vector.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Input/output vector `a`. The result will be written back to this vector.
   * @param vec_b Input vector `b`.
   * @param size Number of elements in the vectors.
   * @param config Configuration for the operation.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_accumulate(T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config);

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

} // namespace icicle