#pragma once

/// @file
/// @brief Types and APIs for the LaBRADOR protocol.

#include "icicle/rings/params/labrador.h" // Zq, Rq, Tq, etc.
#include "icicle/vec_ops.h"               // VecOpsConfig
#include "icicle/negacyclic_ntt.h"        // NegacyclicNTTConfig
#include "icicle/balanced_decomposition.h"
#include "icicle/jl_projection.h"
#include "icicle/norm.h"

// TODO Yuval: cleanup

namespace icicle {
  namespace labrador {

    //------------------------------------------------------------------------------
    // Types
    //------------------------------------------------------------------------------

    using Zq = ::labrador::Zq;
    // Note that Rq/Tq are the same type, left to the user to convert using NTT, and implicit
    // The API is using Rq/Tq or PolyRing for clarity but they are the same type
    using PolyRing = ::labrador::PolyRing; // flat arrays of Zq[d] elements
    using Rq = PolyRing;
    using Tq = PolyRing;

    //------------------------------------------------------------------------------
    // NTT: Rq <-> Tq transforms (negacyclic, d-odd roots of -1)
    //------------------------------------------------------------------------------

    /// @brief Negacyclic NTT/INTT on Rq elements.
    /// Transforms input of length `size` between Rq and Tq using `config`.
    /// @note The transform uses d odd roots of -1, so no zero-padding is needed.
    eIcicleError ntt(const PolyRing* input, int size, NTTDir dir, const NegacyclicNTTConfig& config, PolyRing* output);

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
      Tq* C)
    {
      return eIcicleError::API_NOT_IMPLEMENTED; // TODO integrate
    }

    //------------------------------------------------------------------------------
    // Vector Operations in Zq
    //------------------------------------------------------------------------------

    // TODO Yuval: maybe I need a cpp file to simplify this header
    inline eIcicleError vector_add(const Zq* a, const Zq* b, uint64_t size, const VecOpsConfig& config, Zq* output)
    {
      return icicle::vector_add(a, b, size, config, output);
    }
    inline eIcicleError vector_sub(const Zq* a, const Zq* b, uint64_t size, const VecOpsConfig& config, Zq* output)
    {
      return icicle::vector_sub(a, b, size, config, output);
    }
    inline eIcicleError vector_mul(const Zq* a, const Zq* b, uint64_t size, const VecOpsConfig& config, Zq* output)
    {
      return icicle::vector_mul(a, b, size, config, output);
    }

    //------------------------------------------------------------------------------
    // Vector Operations in Rq/Tq
    //------------------------------------------------------------------------------

    // TODO Yuval: replace Zq vecops with those

    // vectors <Rq/Tq, Rq/Tq> --> Rq/Tq
    eIcicleError
    vector_add(const PolyRing* a, const PolyRing* b, uint64_t size, const VecOpsConfig& config, PolyRing* output);
    eIcicleError
    // vectors of <Tq,Tq> --> Tq
    vector_mul(const Tq* a, const PolyRing* b, uint64_t size, const VecOpsConfig& config, Tq* output);
    eIcicleError
    // vectors of <Rq/Tq,Zq> --> Rq/Tq
    vector_mul(const PolyRing* a, const Zq* b, uint64_t size, const VecOpsConfig& config, PolyRing* output);

    //------------------------------------------------------------------------------
    // Balanced Base Decomposition: Rq -> Rq
    //------------------------------------------------------------------------------

    /// @brief Decompose input in base-b: Z = Z0 + b * Z1.
    /// Used to reduce norm of vectors
    /// Layout: TBD. Output must have elements based on the base.
    // TODO: I am not sure about this API exactly. It is decompoising Rq^n into Rq^nt think but how exactly? it's not
    // digits wise on Zq
    inline eIcicleError decompose(
      const Rq* input, size_t input_size, uint32_t base, const VecOpsConfig& cfg, Rq* output, size_t output_size)
    {
      return icicle::balanced_decomposition::decompose(input, input_size, base, cfg, output, output_size);
    }

    //------------------------------------------------------------------------------
    // Polynomial conjugation
    //------------------------------------------------------------------------------

    /// @brief takes input polynomial p and returns its conjugate polynomial
    Rq conjugate(const Rq& p); // TODO Yuval remove!

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

    /// Returns one or more rows of a JL-matrix, as Rq polynomials, optionally conjugated
    /// TODO: Note in the docs that row_size is measured in Zq
    eIcicleError get_jl_matrix_rows(
      const std::byte* seed,
      size_t seed_len,
      size_t row_size,
      size_t start_row,
      size_t num_rows,
      bool conjugate,
      const VecOpsConfig& cfg,
      Rq* output);

    //------------------------------------------------------------------------------
    // Norm Bounds
    //------------------------------------------------------------------------------

    using icicle::eNormType;

    /// @brief Check whether the norm of a vector is under the bound.
    /// Supports [L2, L∞] norm.
    /// Does the norm_bound have to be uint64_t? What about float? -- could be needed for operatorNorm
    eIcicleError check_norm_bound(
      const Zq* input, size_t size, eNormType norm, uint64_t norm_bound, const VecOpsConfig& cfg, bool* output);

    //------------------------------------------------------------------------------
    // Sampling
    //------------------------------------------------------------------------------

    // TODO integrate with Roman's APIs
    struct SamplingConfig {
      icicleStreamHandle stream = nullptr; ///< Stream for asynchronous execution (if supported).
      bool is_result_on_device =
        false; ///< If true, the output buffer remains on the device (e.g., GPU); otherwise, it's copied to host.
      bool is_async = false;          ///< Whether to launch the sampling asynchronously.
      ConfigExtension* ext = nullptr; ///< Optional backend-specific configuration (e.g., for CUDA/HIP/OpenCL).
    };

    /// @brief Uniform Zq^n sampling.
    /// Used to sample aggregation scalars or projection vectors.
    /// Supports two modes of operation:
    /// - **Slow mode** (`fast_mode = false`): each element is sampled independently via hashing (seed || index).
    /// - **Fast mode** (`fast_mode = true`): one base element is sampled and all others are powers of it.
    eIcicleError random_sampling(
      const std::byte* seed, size_t seed_len, bool fast_mode, const SamplingConfig& cfg, Zq* output, size_t output_size)
    {
      return eIcicleError::API_NOT_IMPLEMENTED; // TODO
    }

    eIcicleError random_sampling(
      const std::byte* seed, size_t seed_len, bool fast_mode, const SamplingConfig& cfg, Tq* output, size_t output_size)
    {
      return random_sampling(seed, seed_len, fast_mode, cfg, (Zq*)output, output_size * Tq::d);
    }

    /// @brief Sample Rq challenge polynomials from challenge space C.
    /// Does not ensure norm constraints (e.g., τ, T) hold. User must check and possibly return with another seed.
    /// seed, seed_len: random seed for sampling and its length
    /// coeff_val = [a1,a2,a3, ..., aN]
    /// num_occur = [m1,m2,m3, ..., mN]
    /// assert(coeff_val.size() == num_occur.size())
    /// Sampling should initialise a polynomial with coefficients consisting of m1 number of a1s, m2 number of a2s, ...,
    /// mN number of aNs. The rest of the coefficients should be 0.
    /// Then, you should randomly flip the signs of the coefficients.
    /// Finally, you need to permute the coefficients randomly

    // TODO Yuval: update this based on Roman's API. Cannot use std::vectors
    eIcicleError sample_challenge_polynomials(
      const std::byte* seed, size_t seed_len, std::vector<size_t> coeff_val, std::vector<size_t> num_occur, Rq output)
    {
      return eIcicleError::API_NOT_IMPLEMENTED; // TODO
    }

  } // namespace labrador
} // namespace icicle