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
    icicleStreamHandle stream; /** Stream for asynchronous execution. */
    bool is_a_on_device;       /** True if `a` is on the device, false if it is not. Default value: false. */
    bool is_b_on_device;       /** True if `b` is on the device, false if it is not. Default value: false. OPTIONAL. */
    bool is_result_on_device;  /** If true, the output is preserved on the device, otherwise on the host. Default value:
                                   false. */
    bool is_async;             /** Whether to run the vector operations asynchronously.
                                   If set to `true`, the function will be non-blocking and synchronization
                                   must be explicitly managed using `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
                                   If set to `false`, the function will block the current CPU thread. */
    int batch_size;            /** Number of vectors (or operations) to process in a batch.
                                   Each vector operation will be performed independently on each batch element.
                                   Default value: 1. */
    bool columns_batch; /** True if the batched vectors are stored as columns in a 2D array (i.e., the vectors are
                           strided in memory as columns of a matrix). If false, the batched vectors are stored
                           contiguously in memory (e.g., as rows or in a flat array). Default value: false. */
    ConfigExtension* ext = nullptr; /** Backend-specific extension. */
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
      1,       // batch_size
      false,   // columns_batch
    };
    return config;
  }

  // Element-wise vector operations

  /**
   * @brief Adds two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_add(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Accumulates the elements of two vectors element-wise and stores the result in the first vector.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first Input/output vector(s). The result will be written back to this vector.
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError
  vector_accumulate(T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config); // use vector_add (inplace)

  /**
   * @brief Subtracts vector `b` from vector `a` element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_sub(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies two vectors element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_mul(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Divides vector `a` by vector `b` element-wise.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the first input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param vec_b Pointer to the second input vector(s).
   *              - The storage layout should match that of `vec_a`.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError vector_div(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Converts elements to and from Montgomery form.
   *
   * @tparam T Type of the elements.
   * @param input Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously in memory.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector.
   * @param is_to_montgomery True to convert into Montgomery form, false to convert out of Montgomery form.
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   *               The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError
  convert_montgomery(const T* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output);

  // Reduction operations

  /**
   * @brief Computes the sum of all elements in each vector in a batch.
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_a Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output array where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   */

  template <typename T>
  eIcicleError vector_sum(const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output);

  /**
   * @brief Computes the product of all elements in each vector in the batch.
   *
   * @tparam T Type of the elements in the vectors.
   * @param vec_a Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector.
   * @param config Configuration for the operation.
   * @param output Pointer to the output array where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   */

  template <typename T>
  eIcicleError vector_product(const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output);

  // Scalar-Vector operations

  /**
   * @brief Adds a scalar to each element of a vector.
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Pointer to the input scalar(s).
   *                 - If `use_single_scalar` is `true`, this should point to a single scalar value.
   *                 - If `use_single_scalar` is `false`, this should point to an array of scalars with length
   * `config.batch_size`.
   * @param vec_b Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in a vector.
   * @param use_single_scalar Flag indicating whether to use a single scalar for all vectors (`true`) or an array of
   * scalars (`false`).
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   * @note To subtract a scalar from each element of a vector - use scalar_add_vec with negative scalar.
   */
  template <typename T>
  eIcicleError scalar_add_vec(
    const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output);

  /**
   * @brief Subtracts each element of a vector from a scalar, elementwise (res[i]=scalar-vec[i]).
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Pointer to Input scalar(s).
   *                 - If `use_single_scalar` is `true`, this should point to a single scalar value.
   *                 - If `use_single_scalar` is `false`, this should point to an array of scalars with length
   * `config.batch_size`.
   * @param vec_b Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in a vector.
   * @param use_single_scalar Flag indicating whether to use a single scalar for all vectors (`true`) or an array of
   * scalars (`false`).
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   * @note To subtract a scalar from each element of a vector - use scalar_add_vec with negative scalar.
   */
  template <typename T>
  eIcicleError scalar_sub_vec(
    const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output);

  /**
   * @brief Multiplies each element of a vector by a scalar.
   *
   * @tparam T Type of the elements in the vector and the scalar.
   * @param scalar_a Pointer to Input scalar(s).
   *                 - If `use_single_scalar` is `true`, this should point to a single scalar value.
   *                 - If `use_single_scalar` is `false`, this should point to an array of scalars with length
   * `config.batch_size`.
   * @param vec_b Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in a vector.
   * @param use_single_scalar Flag indicating whether to use a single scalar for all vectors (`true`) or an array of
   * scalars (`false`).
   * @param config Configuration for the operation.
   * @param output Pointer to the output vector(s) where the results will be stored.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError scalar_mul_vec(
    const T* scalar_a, const T* vec_b, uint64_t size, bool use_single_scalar, const VecOpsConfig& config, T* output);

  // Matrix operations

  /**
   * @brief Transposes a matrix.
   *
   * @tparam T Type of the elements in the matrix.
   * @param mat_in Pointer to the input matrix or matrices.
   * @param nof_rows Number of rows in each input matrix.
   * @param nof_cols Number of columns in each input matrix.
   * @param config Configuration for the operation.
   * @param mat_out Pointer to the output matrix or matrices where the transposed matrices will be stored.
   * @return eIcicleError Error code indicating success or failure.
   * @note The input matrices are assumed to be stored in row-major order.
   *       This function transposes an input matrix or a batch of matrices.
   *       Matrix transpose inplace is not supported for non-power of 2 rows and columns.
   */
  template <typename T>
  eIcicleError
  matrix_transpose(const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out);

  // Miscellaneous operations

  /**
   * @brief Reorders the vector (or batch of vectors) elements based on bit-reverse. That is out[i]=in[bitrev[i]].
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_in Pointer to the input vector(s).
   *              - If `config.batch_size > 1`, this should be a concatenated array of vectors.
   *              - The layout depends on `config.columns_batch`:
   *                - If `false`, vectors are stored contiguously.
   *                - If `true`, vectors are stored as columns in a 2D array.
   * @param size Number of elements in each vector. Must be a power of 2.
   * @param config Configuration for the operation.
   * @param vec_out Pointer to the output vector(s) where the results will be stored.
   *                The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   * @note If `vec_in` and `vec_out` point to the same memory location, the operation is performed in-place.
   */
  template <typename T>
  eIcicleError bit_reverse(const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out);

  /**
   * @brief Extracts a slice from a vector or batch of vectors.
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_in Pointer to the input vector(s).
   * @param offset Offset from which to start the slice in each vector.
   * @param stride Stride between elements in the slice.
   * @param size_in Number of elements in one input vector.
   * @param size_out Number of elements in one input vector.
   * @param config Configuration for the operation.
   * @param vec_out Pointer to the output vector(s) where the results will be stored.
   *                The output array should have the same storage layout as the input vectors.
   * @return eIcicleError Error code indicating success or failure.
   * @note The total input size is `size_in * config.batch_size`.
   *       The total output size is `size_out * config.batch_size`.
   *       parameters must satisfy: offset + (size_out-1) * stride < size_in
   */
  template <typename T>
  eIcicleError slice(
    const T* vec_in,
    uint64_t offset,
    uint64_t stride,
    uint64_t size_in,
    uint64_t size_out,
    const VecOpsConfig& config,
    T* vec_out);

  /**
   * @brief Finds the highest non-zero index in a vector or batch of vectors.
   *
   * @tparam T Type of the elements in the vector.
   * @param vec_in Pointer to the input vector(s).
   * @param size Number of elements in each input vector.
   * @param config Configuration for the operation.
   * @param out_idx Pointer to an array where the output indices of the highest non-zero element in each input vector
   * will be stored. The array should have a length of `config.batch_size`.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename T>
  eIcicleError highest_non_zero_idx(const T* vec_in, uint64_t size, const VecOpsConfig& config, int64_t* out_idx);

  /**
   * @brief Evaluates a polynomial at given domain points.
   *
   * @tparam T Type of the elements in the polynomial and domain.
   * @param coeffs Pointer to the array of coefficients of the polynomial(s).
   *               - The size of `coeffs` should be `coeffs_size * batch_size`.
   *               - If `config.columns_batch` is `false`, coefficients for each polynomial in the batch are stored
   * contiguously.
   *               - If `config.columns_batch` is `true`, coefficients are interleaved.
   * @param coeffs_size Number of coefficients in each polynomial.
   * @param domain Pointer to the array of points at which to evaluate the polynomial(s).
   *               - The same domain is used for all polynomials.
   *               - The size of `domain` should be `domain_size`.
   * @param domain_size Number of domain points.
   * @param config Configuration for the operation.
   * @param evals Pointer to the array where the evaluated results will be stored. This is an output parameter.
   *              - The size of `evals` should be `domain_size * batch_size`.
   *              - If `config.columns_batch` is `false`, results for each polynomial are stored contiguously.
   *              - If `config.columns_batch` is `true`, results are interleaved.
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
   * @brief Divides two polynomials or batch of couples of polynomials.
   *
   * @tparam T Type of the elements in the polynomials.
   * @param numerator Pointer to the array of coefficients of the numerator polynomial(s).
   *                  - The size of `numerator` should be `(numerator_deg + 1) * batch_size`.
   *                  - If `config.columns_batch` is `false`, coefficients for each polynomial in the batch are stored
   * contiguously.
   *                  - If `config.columns_batch` is `true`, coefficients are interleaved.
   * @param numerator_deg Degree of the numerator polynomial.
   * @param denominator Pointer to the array of coefficients of the denominator polynomial(s).
   *                  - Storage layout is similar to `numerator`.
   * @param denominator_deg Degree of the denominator polynomial.
   * @param config Configuration for the operation.
   * @param q_size Size of the quotient array for one polynomial.
   * @param r_size Size of the remainder array.
   * @param q_out Pointer to the array where the quotient polynomial(s) will be stored. This is an output parameter.
   *              - The storage layout should match that of `numerator`.
   * @param r_out Pointer to the array where the remainder polynomial(s) will be stored. This is an output parameter.
   *              - The storage layout should match that of `numerator`.
   *              - The size of `r_out` should be sufficient to hold the remainder coefficients for each polynomial.
   * @return eIcicleError Error code indicating success or failure.
   *
   * @note The degrees should satisfy `numerator_deg >= denominator_deg`.
   *       The sizes `q_size` and `r_size` must be at least `numerator_deg - denominator_deg + 1` and `denominator_deg`,
   * respectively. The function assumes that the input and output arrays are properly allocated.
   */
  template <typename T>
  eIcicleError polynomial_division(
    const T* numerator,
    int64_t numerator_deg,
    const T* denumerator,
    int64_t denumerator_deg,
    uint64_t q_size,
    uint64_t r_size,
    const VecOpsConfig& config,
    T* q_out /*OUT*/,
    T* r_out /*OUT*/);

} // namespace icicle