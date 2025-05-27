#pragma once

/// @file
/// @brief Types and APIs for the LaBRADOR protocol.

#include "icicle/rings/integer_rings/labrador.h" // Zq, Rq, Tq, etc.
#include "icicle/vec_ops.h"                      // VecOpsConfig
#include "icicle/ntt.h"                          // NTTConfig
#include "icicle/random_sampling.h"              // Random sampling

namespace icicle {
  namespace labrador {

    //------------------------------------------------------------------------------
    // Types
    //------------------------------------------------------------------------------

    using Zq = ::labrador::Zq;
    // using ZqRns = labrador::ZqRns; // TODO: Consider RNS later for performance
    using Rq = ::labrador::Rq; // Rq: flat arrays of Zq[d] elements
    using Tq = ::labrador::Tq; // Tq: flat arrays of Zq[d] elements, typically in NTT domain

    //------------------------------------------------------------------------------
    // NTT: Rq <-> Tq transforms (negacyclic, d-odd roots of -1)
    //------------------------------------------------------------------------------

    /// @brief Negacyclic NTT/INTT on Rq elements.
    /// Transforms input of length `size` between Rq and Tq using `config`.
    /// @note The transform uses d odd roots of -1, so no zero-padding is needed.
    eIcicleError ntt(const Zq* input, int size, NTTDir dir, const NTTConfig<Zq>& config, Zq* output);

    //------------------------------------------------------------------------------
    // Matrix Multiplication in Tq (e.g., Ajtai commitments)
    //------------------------------------------------------------------------------

    /// @brief Matrix multiplication in Tq.
    /// Computes C = A * B for Tq matrices.
    /// @note Batching is not supported. Output buffer must be sized (A_rows x B_cols).
    eIcicleError matmul(
      const Tq* A,
      uint32_t A_nof_rows,
      uint32_t A_nof_cols,
      const Tq* B,
      uint32_t B_nof_rows,
      uint32_t B_nof_cols,
      const VecOpsConfig& config,
      Tq* C);

    // TODO: Add Ajtai matmul with seeded A/B to avoid storing large matrices?

    //------------------------------------------------------------------------------
    // Vector Operations in Zq
    //------------------------------------------------------------------------------

    eIcicleError vector_add(const Zq* a, const Zq* b, uint64_t size, const VecOpsConfig& config, Zq* output);
    eIcicleError vector_sub(const Zq* a, const Zq* b, uint64_t size, const VecOpsConfig& config, Zq* output);
    eIcicleError vector_mul(const Zq* a, const Zq* b, uint64_t size, const VecOpsConfig& config, Zq* output);

    // TODO: need scalar-vector ops for Rq or Zq?

    //------------------------------------------------------------------------------
    // Balanced Base Decomposition: Rq -> Rq
    //------------------------------------------------------------------------------

    /// @brief Decompose input in base-b: Z = Z0 + b * Z1.
    /// Used to reduce norm of vectors
    /// Layout: TBD. Output must have elements based on the base.
    // TODO: I am not sure about this API exactly. It is decompoising Rq^n into Rq^nt think but how exactly? it's not
    // digits wise on Zq
    eIcicleError decompose(
      const Rq* input, size_t input_size, uint32_t base, const VecOpsConfig& cfg, Rq* output, size_t output_size);

    //------------------------------------------------------------------------------
    // Polynomial conjugation
    //------------------------------------------------------------------------------

    /// @brief takes input polynomial p and returns its conjugate polynomial
    Rq conjugate(const Rq& p);
    //------------------------------------------------------------------------------
    // Johnson–Lindenstrauss Projection
    //------------------------------------------------------------------------------

    /// @brief JL projection from Zqⁿ to Zqᵐ using a seeded pseudo-random matrix.
    /// Used to compress witness vector Si (flat Zq array) into lower dimension.
    eIcicleError jl_projection(
      const Zq* input,
      size_t input_size,
      const std::byte* seed,
      size_t seed_len,
      const VecOpsConfig& cfg,
      Zq* output,
      size_t output_size);

    /// Helper function to get a single row from JL projection matrix
    eIcicleError get_jl_matrix_row(
      const std::byte* seed,
      size_t seed_len,
      size_t matrix_rows, // N parameter (256 in our case)
      size_t matrix_cols, // M parameter (n*d in our case)
      size_t row_index,   // which row to extract
      const VecOpsConfig& cfg,
      Zq* output) // output array of size matrix_cols
    {
      return eIcicleError::SUCCESS;
    }

    //------------------------------------------------------------------------------
    // Norm Bounds
    //------------------------------------------------------------------------------

    enum class eNormType {
      L2 = 0,    ///< Euclidean norm: sqrt(sum of squares)
      LInfinity, ///< Max-norm: max(abs())
      Lop        ///< Operator norm (used in challenge space)
    };

    /// @brief Check whether the norm of a vector is under the bound.
    /// Supports L2, L∞, or Operator norm.
    eIcicleError check_norm_bound(
      const Zq* input, size_t size, eNormType norm, uint64_t norm_bound, const VecOpsConfig& cfg, bool* output);

    //------------------------------------------------------------------------------
    // Sampling
    //------------------------------------------------------------------------------

    /// @brief Uniform Zq^n sampling.
    /// Used to sample aggregation scalars or projection vectors.
    /// Supports two modes of operation:
    /// - **Slow mode** (`fast_mode = false`): each element is sampled independently via hashing (seed || index).
    /// - **Fast mode** (`fast_mode = true`): one base element is sampled and all others are powers of it.
    eIcicleError random_sampling(
      const std::byte* seed,
      size_t seed_len,
      bool fast_mode,
      const SamplingConfig& cfg,
      Zq* output,
      size_t output_size);

    /// @brief Sample Rq challenge polynomials from challenge space C.
    /// Does not ensure norm constraints (e.g., τ, T) hold. User must check and possibly return with another seed.
    eIcicleError sample_challenge_polynomials(
      const std::byte* seed,
      size_t seed_len,
      bool fast_mode,
      const SamplingConfig& cfg,
      Rq* output,
      size_t output_size);

    //------------------------------------------------------------------------------
    // TODO / Notes
    //------------------------------------------------------------------------------

    // TODO: Add seeded matrix generators for amortized Ajtai/Avec commitments
    //       to reduce memory in proving large witness vectors.

  } // namespace labrador
} // namespace icicle