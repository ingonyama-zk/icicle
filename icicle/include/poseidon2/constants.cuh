#pragma once
#ifndef POSEIDON2_CONSTANTS_H
#define POSEIDON2_CONSTANTS_H

#include "gpu-utils/device_context.cuh"

namespace poseidon2 {
  /**
   * For most of the Poseidon2 configurations this is the case
   */
  const int EXTERNAL_ROUNDS_DEFAULT = 8;

  enum DiffusionStrategy {
    DEFAULT_DIFFUSION,
    MONTGOMERY,
  };

  enum MdsType { DEFAULT_MDS, PLONKY };

  /**
   * @struct Poseidon2Constants
   * This constants are enough to define a Poseidon2 instantce
   * @param round_constants A pointer to round constants allocated on the device
   * @param mds_matrix A pointer to an mds matrix allocated on the device
   * @param non_sparse_matrix A pointer to non sparse matrix allocated on the device
   * @param sparse_matrices A pointer to sparse matrices allocated on the device
   */
  template <typename S>
  struct Poseidon2Constants {
    int width;
    int alpha;
    int internal_rounds;
    int external_rounds;
    S* round_constants = nullptr;
    S* internal_matrix_diag = nullptr;
    MdsType mds_type;
    DiffusionStrategy diffusion;
  };

  template <typename S>
  cudaError_t create_poseidon2_constants(
    int width,
    int alpha,
    int internal_rounds,
    int external_rounds,
    const S* round_constants,
    const S* internal_matrix_diag,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<S>* poseidon_constants);

  template <typename S>
  cudaError_t init_poseidon2_constants(
    int width,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<S>* poseidon2_constants);

  template <typename S>
  cudaError_t release_poseidon2_constants(Poseidon2Constants<S>* constants, device_context::DeviceContext& ctx);
} // namespace poseidon2

#endif