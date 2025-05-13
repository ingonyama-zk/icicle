#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {

  /// @brief Performs Johnson–Lindenstrauss (JL) projection on an input vector.
  ///
  /// Projects an input vector of length N to a fixed output vector of length 256
  /// using a pseudo-random projection matrix with entries in {-1, 0, 1}.
  /// The matrix is not stored explicitly; instead, each column is deterministically
  /// generated on-demand from a cryptographic hash:
  ///
  ///     column_i = decode(Hash(seed || i))
  ///
  /// Each hash output is 512 bits, interpreted as 256 2-bit pairs, each mapped to
  /// a matrix entry according to:
  ///     00 →  0
  ///     01 →  1
  ///     10 → -1
  ///     11 →  0
  ///
  /// This yields the correct probability distribution:
  ///     Pr[0] = 0.5, Pr[1] = 0.25, Pr[-1] = 0.25
  ///
  /// The JL projection is computed as a sum of weighted columns:
  ///     output = ∑_{i=0}^{N-1} input[i] * column_i
  ///
  /// The implementation involves only additions and subtractions (no multiplications),
  /// and is designed to be efficient and parallel-friendly.
  ///
  /// @tparam T         Element type (e.g., Zq)
  /// @param input      Pointer to input vector (length = N)
  /// @param N          Length of the input vector
  /// @param seed       Pointer to seed for deterministic hash-based sampling
  /// @param seed_len   Length of the seed buffer in bytes
  /// @param cfg        Vector operation configuration (e.g., backend, batching)
  /// @param output     Pointer to output buffer (length = 256 elements)
  /// @return           eIcicleError::SUCCESS on success, or an appropriate error code
  template <typename T>
  eIcicleError jl_projection(
    const T* input,
    size_t N,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& cfg,
    T* output // length 256
  );

} // namespace icicle