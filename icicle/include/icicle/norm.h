#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {

  /**
   * @brief Enum to represent supported norm types for vectors.
   */
  enum class eNormType {
    L2 = 0,   ///< Euclidean norm: sqrt(sum of squares)
    LInfinity ///< Max norm: maximum absolute element value
  };

  namespace norm {
    /**
     * @brief Checks whether the norm of a vector is within a specified bound.
     * @note This function assumes that:
     * - Each element in the input vector is at most sqrt(q) in magnitude
     * - The vector size is at most 2^16 elements
     * If these assumptions are violated, the function will return eIcicleError::INVALID_ARGUMENT
     *
     * @tparam T             Element type of the input vector (e.g., int64_t, uint64_t)
     * @param input          Pointer to the input vector
     * @param size           Number of elements in the vector
     * @param norm           Type of norm to compute (L2 or LInfinity)
     * @param norm_bound     Upper bound on the norm value
     * @param cfg            Configuration for vector operations (e.g., threading, SIMD, batching)
     * @param output         Pointer to a boolean; set to true if norm is within bound, false otherwise
     * @return               eIcicleError::SUCCESS if check was successful,
     *                       eIcicleError::INVALID_ARGUMENT or other error codes on failure
     */
    template <typename T>
    eIcicleError check_norm_bound(
      const T* input, size_t size, eNormType norm, uint64_t norm_bound, const VecOpsConfig& cfg, bool* output);

    /**
     * @brief Checks whether norm(a) < scalar:u64 * norm(b)
     *
     * This is useful in lattice-based schemes and other relative-norm comparisons
     * where an exact bound is not known in advance but depends on a second input vector.
     * @note This function assumes that:
     * - Each element in the input vectors is at most sqrt(q) in magnitude
     * - The vector size is at most 2^16 elements
     * If these assumptions are violated, the function will return eIcicleError::INVALID_ARGUMENT
     *
     * @tparam T             Element type of the input vectors
     * @param input_a        Pointer to the input vector whose norm is being checked
     * @param input_b        Pointer to the comparison vector whose norm is scaled by `scalar_b`
     * @param size           Number of elements in each vector
     * @param norm           Type of norm to compute (L2 or LInfinity)
     * @param scale          Scaling multiplier to apply to norm(input_b)
     * @param cfg            Configuration for vector operations (e.g. batching)
     * @param output         Pointer to a boolean; set to true if norm(input_a) < scalar * norm(input_b)
     * @return               eIcicleError::SUCCESS if check was successful,
     *                       eIcicleError::INVALID_ARGUMENT or other error codes on failure
     */
    template <typename T>
    eIcicleError check_norm_relative(
      const T* input_a,
      const T* input_b,
      size_t size,
      eNormType norm,
      uint64_t scale,
      const VecOpsConfig& cfg,
      bool* output);

  } // namespace norm

} // namespace icicle