#pragma once

#include "icicle/vec_ops.h" // For VecOpsConfig
#include "icicle/hash.h"    // For cryptographic hashing utilities

namespace icicle {

  /**
   * @brief Configuration for random sampling operations.
   */
  struct SamplingConfig {
    icicleStreamHandle stream = nullptr; ///< Stream for asynchronous execution (if supported).
    bool is_result_on_device =
      false; ///< If true, the output buffer remains on the device (e.g., GPU); otherwise, it's copied to host.
    bool is_async = false;          ///< Whether to launch the sampling asynchronously.
    ConfigExtension* ext = nullptr; ///< Optional backend-specific configuration (e.g., for CUDA/HIP/OpenCL).
  };

  /**
   * @brief Samples `output_size` elements of type `T` using a deterministic hash-based PRNG seeded with `seed`.
   *
   * Supports two modes of operation:
   * - **Slow mode** (`fast_mode = false`): each element is sampled independently via hashing (seed || index).
   * - **Fast mode** (`fast_mode = true`): one base element is sampled and all others are powers of it.
   *
   * Typical use cases:
   * - `T = Zq`: individual field elements.
   * - `T = Rq`: polynomials with coefficients in Zq (internally reduced to a call over Zq).
   * - `T = Tq`: alias for Rq.
   *
   * @note For composite types like Rq/Tq, this function simply invokes the Zq version repeatedly,
   *       since ICICLE only requires backend implementations for scalar Zq elements.
   *
   * @tparam T             Type of elements to sample (Zq, Rq, Tq, etc.)
   * @param seed           Pointer to the seed buffer
   * @param seed_len       Length of the seed buffer in bytes
   * @param fast_mode      Enables optimized sampling using repeated powers if true
   * @param cfg            Sampling configuration (stream, async, output location, etc.)
   * @param output         Pointer to output buffer of type T
   * @param output_size    Number of elements to generate
   * @return               eIcicleError::SUCCESS on success, or appropriate error code
   */
  template <typename T>
  eIcicleError random_sampling(
    const std::byte* seed, size_t seed_len, bool fast_mode, const SamplingConfig& cfg, T* output, size_t output_size);

  // NOTE:
  // This template is meant to be reused for any type T where sampling is defined.
  // For polynomial types (e.g., Rq, Tq), the implementation should simply call the Zq
  // backend implementation for each coefficient. ICICLE only needs a backend
  // implementation for Zq; higher-level types are handled via templated wrappers.

  // Random-challenge polynomial sampling
  // TODO: this needs to take tau1, tau2
  template <typename T>
  eIcicleError random_challenge_polynomial(
    const std::byte* seed, size_t seed_len, const SamplingConfig& cfg, T* output, size_t output_size);

} // namespace icicle