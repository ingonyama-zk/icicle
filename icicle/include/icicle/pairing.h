#pragma once

#include <functional>

#include "errors.h"
#include "runtime.h"

namespace icicle {
  /**
   * @brief Computes the pairing
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
  template <typename A, typename A2, typename Pairing, typename TargetField = typename Pairing::target_field_t>
  eIcicleError pairing(const A& p, const A2& q, TargetField* output);
} // namespace icicle