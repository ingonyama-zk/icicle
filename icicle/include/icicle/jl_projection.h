#pragma once

#include "errors.h"
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
  /// @param input         Input vector (length = input_size)
  /// @param input_size    Number of elements in the input vector
  /// @param seed          Seed used for deterministic JL matrix row generation
  /// @param seed_len      Length of seed in bytes
  /// @param cfg           Vector operation configuration
  /// @param output        Output vector (length = output_size)
  /// @param output_size   Number of projection rows
  /// @return              eIcicleError::SUCCESS on success
  template <typename T>
  eIcicleError jl_projection(
    const T* input,
    size_t input_size,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& cfg,
    T* output,
    size_t output_size);

  /// @brief Generates raw JL matrix rows for scalar elements (e.g., Zq).
  ///
  /// Each row is length `row_size`, with values in {-1, 0, 1} encoded in Zq.
  /// Output is row-major and laid out as [row_0 | row_1 | ...].
  ///
  /// @tparam T            Element type (e.g., Zq)
  /// @param seed          Pointer to seed used for deterministic row generation
  /// @param seed_len      Length of the seed buffer in bytes
  /// @param row_size      Number of elements per row
  /// @param start_row     Index of the first row to generate
  /// @param num_rows      Number of rows to generate
  /// @param cfg           Vector operation configuration
  /// @param output        Output buffer (size = num_rows × row_size)
  /// @return              eIcicleError::SUCCESS on success
  template <typename T>
  eIcicleError get_jl_matrix_rows(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    const VecOpsConfig& cfg,
    T* output);

  /// @brief Generates JL matrix rows grouped as Rq polynomials, optionally conjugated.
  ///
  /// Each JL row is interpreted as a sequence of `row_size` Rq polynomials.
  /// The Rq type is expected to define a static member `Rq::d`, its degree.
  ///
  /// If `conjugate == true`, each polynomial is laid out to match:
  ///     a(X) ↦ a(X⁻¹) mod (X^d + 1)
  ///
  /// This API is useful for preparing structured JL matrices for constraint systems
  /// where each row operates over multiple `Rq` values rather than scalar `Zq` entries.
  ///
  /// Output is laid out in row-major order: output[i * row_size + j] = Rq_j in row i.
  ///
  /// @tparam Rq           Polynomial ring type (must define static constexpr size_t d)
  /// @param seed          Seed used for deterministic JL matrix generation
  /// @param seed_len      Length of the seed buffer in bytes
  /// @param row_size      Number of Rq polynomials per row
  /// @param start_row     Index of the first JL row to generate
  /// @param num_rows      Number of rows to generate
  /// @param conjugate     If true, polynomials are returned in conjugated layout
  /// @param cfg           Vector operation configuration
  /// @param output        Output buffer (length = num_rows * row_size)
  /// @return              eIcicleError::SUCCESS on success
  template <typename Rq>
  eIcicleError get_jl_matrix_rows(
    const std::byte* seed,
    size_t seed_len,
    size_t row_size,
    size_t start_row,
    size_t num_rows,
    bool conjugate,
    const VecOpsConfig& cfg,
    Rq* output);

} // namespace icicle