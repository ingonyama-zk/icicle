#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig

namespace icicle {

  /// @brief Performs Johnson–Lindenstrauss (JL) projection on an input vector.
  ///
  /// Projects an input vector of length N to a fixed lower-dimensional output vector
  /// of length `output_size` (commonly 256) using a pseudo-random projection matrix
  /// with entries in {-1, 0, 1}.
  ///
  /// The projection matrix is not stored explicitly. Instead, each **row** of the matrix
  /// is generated on-the-fly using a cryptographic hash:
  ///
  ///     chunk = decode(Hash(seed || counter))
  ///
  /// where `counter = row_index * num_chunks_per_row + chunk_index`.
  /// Each 512-bit hash output yields 256 2-bit pairs, interpreted as:
  ///     00 →  0
  ///     01 →  1
  ///     10 → -1
  ///     11 →  0
  ///
  /// This results in the correct probability distribution for JL projection:
  ///     Pr[0] = 0.5, Pr[1] = 0.25, Pr[-1] = 0.25
  ///
  /// Each output element is computed as the inner product of a pseudo-random row with the input vector:
  ///     output[i] = ∑_{j=0}^{N-1} A[i][j] * input[j]
  ///
  /// where A[i][j] ∈ {-1, 0, 1} is derived from the hash-based row generation.
  ///
  /// The implementation avoids multiplications (only additions and subtractions),
  /// is memory-efficient (no matrix stored), and designed to support parallel row-wise execution.
  ///
  /// @tparam T            Element type (e.g., Zq)
  /// @param input         Pointer to input vector (length = input_size)
  /// @param input_size    Length of the input vector
  /// @param seed          Pointer to seed used for deterministic hash-based sampling
  /// @param seed_len      Length of the seed buffer in bytes
  /// @param cfg           Vector operation configuration (e.g., backend, batching)
  /// @param output        Pointer to output buffer (length = output_size)
  /// @param output_size   Number of projection rows (i.e., reduced dimension)
  /// @return              eIcicleError::SUCCESS on success, or an appropriate error code
  template <typename T>
  eIcicleError jl_projection(
    const T* input,
    size_t input_size,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& cfg,
    T* output,
    size_t output_size);

  // TODO Yuval: try std::span

} // namespace icicle