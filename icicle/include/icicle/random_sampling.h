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
} // namespace icicle