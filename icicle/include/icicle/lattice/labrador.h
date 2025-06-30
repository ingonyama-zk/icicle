#pragma once

/// @file
/// @brief High-level LaBRADOR APIs and types: NTT, vector ops, JL projection, norm checks, and sampling.

#include "icicle/rings/params/labrador.h" // Zq, Rq, Tq, etc.

// TODO(Yuval): Move implementation to source file and reduce header dependencies
#include "icicle/vec_ops.h"
#include "icicle/negacyclic_ntt.h"
#include "icicle/balanced_decomposition.h"
#include "icicle/jl_projection.h"
#include "icicle/norm.h"
#include "icicle/hash/keccak.h" // For Hash

namespace icicle {
  namespace labrador {

    //------------------------------------------------------------------------------
    // Type Aliases
    //------------------------------------------------------------------------------

    using Zq = ::labrador::Zq;
    using PolyRing = ::labrador::PolyRing; // Flat arrays of Zq[d]; used as both Rq and Tq
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
      const VecOpsConfig& config,
      Tq* C)
    {
      return icicle::matmul(A, A_nof_rows, A_nof_cols, B, B_nof_rows, B_nof_cols, config, C);
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
    /// TODO: Note in the docs that row_size is measured in Zq. TODO Ash: we can make it be in Rq too. Let me know
    /// please
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

    /// @brief Uniform Zq^n sampling.
    /// Used to sample aggregation scalars or projection vectors.
    /// Supports two modes of operation:
    /// - **Slow mode** (`fast_mode = false`): each element is sampled independently via hashing (seed || index).
    /// - **Fast mode** (`fast_mode = true`): one base element is sampled and all others are powers of it.
    inline eIcicleError random_sampling(
      const std::byte* seed, size_t seed_len, bool /*fast_mode*/, const SamplingConfig&, Zq* output, size_t output_size)
    {
      // SHA3-256 based deterministic PRNG seeded with (seed || index)
      icicle::Hash hasher = icicle::Sha3_256::create();

      std::vector<std::byte> buf(seed, seed + seed_len);
      buf.resize(seed_len + 8); // reserve room for the index

      // modulus q  (fits in 64 bits, see balanced_decomposition.h)
      constexpr auto q_storage = Zq::get_modulus();
      const uint64_t q = *reinterpret_cast<const uint64_t*>(&q_storage);

      for (size_t i = 0; i < output_size; ++i) {
        std::memcpy(buf.data() + seed_len, &i, 8); // append index (LE)
        std::array<std::byte, 32> digest{};
        hasher.hash(buf.data(), buf.size(), {}, digest.data());

        uint64_t word;
        std::memcpy(&word, digest.data(), sizeof(word));
        output[i] = Zq::from(word % q); // uniform in Z_q
      }
      return eIcicleError::SUCCESS;
    }

    inline eIcicleError random_sampling(
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
    inline eIcicleError sample_challenge_polynomials(
      const std::byte* seed, size_t seed_len, std::vector<size_t> coeff_val, std::vector<size_t> num_occur, Rq output)
    {
      return eIcicleError::API_NOT_IMPLEMENTED; // TODO
    }

  } // namespace labrador
} // namespace icicle