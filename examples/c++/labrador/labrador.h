#pragma once

/// @file
/// @brief High-level LaBRADOR APIs and types: NTT, vector ops, JL projection, norm checks, and sampling.

#include "icicle/rings/params/babykoala.h" // Zq, Rq, Tq, etc.

// TODO(Yuval): Move implementation to source file and reduce header dependencies
#include "icicle/vec_ops.h"
#include "icicle/mat_ops.h"
#include "icicle/negacyclic_ntt.h"
#include "icicle/balanced_decomposition.h"
#include "icicle/jl_projection.h"
#include "icicle/norm.h"
#include "icicle/hash/keccak.h" // For Hash
#include "icicle/random_sampling.h"
#include "examples_utils.h"

namespace icicle {
  namespace labrador {

    //------------------------------------------------------------------------------
    // Type Aliases
    //------------------------------------------------------------------------------

    using Zq = ::babykoala::Zq;
    using PolyRing = ::babykoala::PolyRing; // Flat arrays of Zq[d]; used as both Rq and Tq
    using Rq = PolyRing;
    using Tq = PolyRing;

    //------------------------------------------------------------------------------
    // NTT: Rq <-> Tq (Negacyclic, d-odd roots of -1)
    //------------------------------------------------------------------------------

    /// @brief Negacyclic NTT or INTT on Rq elements.
    inline eIcicleError
    ntt(const PolyRing* input, int size, NTTDir dir, const NegacyclicNTTConfig& config, PolyRing* output)
    {
      return icicle::ntt(input, size, dir, config, output);
    }

    //------------------------------------------------------------------------------
    // Matrix Multiplication in Tq
    //------------------------------------------------------------------------------

    /// @brief Compute matrix product C = A * B in Tq.
    /// @note No batching supported. Output buffer must be (A_rows × B_cols).
    inline eIcicleError matmul(
      const Tq* A,
      uint32_t A_nof_rows,
      uint32_t A_nof_cols,
      const Tq* B,
      uint32_t B_nof_rows,
      uint32_t B_nof_cols,
      const MatMulConfig& config,
      Tq* C)
    {
      return icicle::matmul(A, A_nof_rows, A_nof_cols, B, B_nof_rows, B_nof_cols, config, C);
    }

    inline eIcicleError matrix_transpose(
      const PolyRing* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, PolyRing* mat_out)
    {
      return icicle::matrix_transpose<PolyRing>(mat_in, nof_rows, nof_cols, config, mat_out);
    }

    //------------------------------------------------------------------------------
    // Vector Operations: Add/Mul in Rq/Tq
    //------------------------------------------------------------------------------

    inline eIcicleError
    vector_add(const PolyRing* a, const PolyRing* b, uint64_t size, const VecOpsConfig& config, PolyRing* output)
    {
      return icicle::vector_add(a, b, size, config, output);
    }

    inline eIcicleError vector_mul(const Tq* a, const Tq* b, uint64_t size, const VecOpsConfig& config, Tq* output)
    {
      return icicle::vector_mul(a, b, size, config, output);
    }

    inline eIcicleError
    vector_mul(const PolyRing* a, const Zq* b, uint64_t size, const VecOpsConfig& config, PolyRing* output)
    {
      return icicle::vector_mul(a, b, size, config, output);
    }

    // TODO: Add optional reduction/sum if required by protocol

    //------------------------------------------------------------------------------
    // Balanced Decomposition: Rq -> Rq (digit-major)
    //------------------------------------------------------------------------------

    /// @brief Decompose input in base-b: Z = Z₀ + b·Z₁ + ...
    /// Output layout is digit-major (first all Z₀ polynomials, then Z₁, ...).
    inline eIcicleError decompose(
      const Rq* input, size_t input_size, uint32_t base, const VecOpsConfig& cfg, Rq* output, size_t output_size)
    {
      return icicle::balanced_decomposition::decompose<Rq>(input, input_size, base, cfg, output, output_size);
    }

    inline eIcicleError recompose(
      const Rq* input, size_t input_size, uint32_t base, const VecOpsConfig& cfg, Rq* output, size_t output_size)
    {
      return icicle::balanced_decomposition::recompose<Rq>(input, input_size, base, cfg, output, output_size);
    }

    //------------------------------------------------------------------------------
    // Johnson–Lindenstrauss Projection
    //------------------------------------------------------------------------------

    /// @brief JL projection from Zqⁿ to Zqᵐ using seeded pseudo-random matrix.
    inline eIcicleError jl_projection(
      const Zq* input,
      size_t input_size,
      const std::byte* seed,
      size_t seed_len,
      const VecOpsConfig& cfg,
      Zq* output,
      size_t output_size)
    {
      return icicle::jl_projection<Zq>(input, input_size, seed, seed_len, cfg, output, output_size);
    }

    /// Returns one or more rows of a JL-matrix, as Rq polynomials, optionally conjugated
    inline eIcicleError get_jl_matrix_rows(
      const std::byte* seed,
      size_t seed_len,
      size_t row_size,
      size_t start_row,
      size_t num_rows,
      bool conjugate,
      const VecOpsConfig& cfg,
      Rq* output)
    {
      return icicle::get_jl_matrix_rows(seed, seed_len, row_size, start_row, num_rows, conjugate, cfg, output);
    }

    //------------------------------------------------------------------------------
    // Norm Bounds
    //------------------------------------------------------------------------------

    using icicle::eNormType;

    /// @brief Check whether the norm of a vector is under the bound.
    /// Supports [L2, L∞] norm.
    /// Does the norm_bound have to be uint64_t? What about float? -- could be needed for operatorNorm
    inline eIcicleError check_norm_bound(
      const Zq* input, size_t size, eNormType norm, uint64_t norm_bound, const VecOpsConfig& cfg, bool* output)
    {
      return icicle::norm::check_norm_bound<Zq>(input, size, norm, norm_bound, cfg, output);
    }

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

    inline eIcicleError random_sampling(
      size_t output_size, bool fast_mode, const std::byte* seed, size_t seed_len, const VecOpsConfig& cfg, Zq* output)
    {
      // TODO: forced on CPU
      ScopedCpuDevice force_cpu{};
      return icicle::random_sampling(output_size, fast_mode, seed, seed_len, cfg, output);
    }

    /// @brief Uniform Zq^n sampling.
    /// Used to sample aggregation scalars or projection vectors.
    /// Supports two modes of operation:
    /// - **Slow mode** (`fast_mode = false`): each element is sampled independently via hashing (seed || index).
    /// - **Fast mode** (`fast_mode = true`): one base element is sampled and all others are powers of it.
    inline eIcicleError random_sampling(
      size_t output_size, bool fast_mode, const std::byte* seed, size_t seed_len, const VecOpsConfig& cfg, Tq* output)
    {
      // TODO: forced on CPU
      ScopedCpuDevice force_cpu{};
      return icicle::random_sampling(output_size * Tq::d, fast_mode, seed, seed_len, cfg, (Zq*)output);
    }

    /// TODO update:
    /// @brief Sample Rq challenge polynomials from challenge space C.
    inline eIcicleError sample_challenge_space_polynomials(
      const std::byte* seed,
      size_t seed_len,
      size_t size,
      uint32_t ones,
      uint32_t twos,
      uint64_t norm,
      const VecOpsConfig& config,
      Rq* output)
    {
      return icicle::sample_challenge_space_polynomials(seed, seed_len, size, ones, twos, norm, config, output);
    }

  } // namespace labrador
} // namespace icicle