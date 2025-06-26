#pragma once

#include "icicle/errors.h"
#include "icicle/vec_ops.h"

namespace icicle {
  /**
   * @brief Randomly sample elements of type T from a seeded uniform distribution.
   *
   * This function fills the output buffer with random elements of type T, using the provided seed and configuration.
   * The sampling can be performed in fast (non-cryptographic) or secure mode.
   *
   * @tparam T The element type to sample (e.g., field or ring).
   * @param size Number of elements to sample.
   * @param fast_mode If true, use fast (non-cryptographic) sampling; otherwise, use secure sampling.
   * @param seed Pointer to the seed buffer for deterministic sampling.
   * @param seed_len Length of the seed buffer in bytes.
   * @param config Vector operations configuration (e.g., backend, device).
   * @param output Output buffer to store the sampled elements (must have at least 'size' elements).
   * @return eIcicleError::SUCCESS on success, or an error code on failure.
   */
  template <typename T>
  eIcicleError random_sampling(
    size_t size, bool fast_mode, const std::byte* seed, size_t seed_len, const VecOpsConfig& config, T* output);

  /**
   * @brief Sample Rq challenge polynomials with {0, 1, 2, -1, -2} coefficients.
   *
   * This function samples challenge polynomials with specific coefficient patterns. The sampling process:
   * 1. Initializes a polynomial with coefficients consisting of "ones" number of ±1s, "twos" number of ±2s, and the rest
   * of the coefficients are 0s.
   * 2. Randomly flips the signs of the coefficients
   * 3. Permutes the coefficients randomly
   *
   * Note: This function does not ensure norm constraints (e.g., τ, T) hold
   *
   * @tparam T The element type for the polynomial coefficients.
   * @param seed Pointer to the seed buffer for deterministic sampling.
   * @param seed_len Length of the seed buffer in bytes.
   * @param size Number of polynomials to sample.
   * @param ones Number of 1s coefficients in each polynomial.
   * @param twos Number of 2s coefficients in each polynomial.
   * @param output Output buffer to store the sampled Rq polynomials. Should be of size config.batch_size.
   * @return eIcicleError::SUCCESS on success, or an error code on failure.
   *
   */
  template <typename T>
  eIcicleError sample_challenge_space_polynomials(
    const std::byte* seed,
    size_t seed_len,
    size_t size,
    uint32_t ones,
    uint32_t twos,
    const VecOpsConfig& config,
    T* output);
} // namespace icicle