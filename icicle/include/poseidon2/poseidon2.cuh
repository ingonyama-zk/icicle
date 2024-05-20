#pragma once
#ifndef POSEIDON2_H
#define POSEIDON2_H

#include <cstdint>
#include <stdexcept>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"
#include "hash/hash.cuh"

/**
 * @namespace poseidon2
 * Implementation of the [Poseidon2 hash function](https://eprint.iacr.org/2019/458.pdf)
 * Specifically, the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/)
 */
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

  /**
   * Loads pre-calculated optimized constants, moves them to the device
   */
  template <typename S>
  cudaError_t init_poseidon2_constants(
    int width,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<S>* constants);

  template <typename S>
  cudaError_t release_poseidon2_constants(Poseidon2Constants<S>* constants, device_context::DeviceContext& ctx);

  template <typename S>
  class Poseidon2 : public Permutation<S>,
                    public CompressionHasher<S>
  {
    Poseidon2Constants<S> constants;

    cudaError_t squeeze_states(
      const S* states,
      unsigned int number_of_states,
      unsigned int rate,
      S* output,
      device_context::DeviceContext& ctx,
      unsigned int offset=0
    );

    cudaError_t run_permutation_kernel(
          const S* states,
          S* output,
          unsigned int number_of_states,
          device_context::DeviceContext& ctx
    );

  public: 
    Poseidon2(Poseidon2& other) : constants(other.constants) {}
    Poseidon2(Poseidon2Constants<S> constants) : constants(constants) {}
    Poseidon2(int width, MdsType mds_type, DiffusionStrategy diffusion, device_context::DeviceContext& ctx);
    ~Poseidon2();

    cudaError_t permute_many(
        const S* states,
        S* output,
        unsigned int number_of_states,
        device_context::DeviceContext& ctx
    ) override;

    cudaError_t compress_many(
        const S* states,
        S* output,
        unsigned int number_of_states,
        unsigned int rate,
        device_context::DeviceContext& ctx,
        unsigned int offset=0,
        S* perm_output=nullptr
    ) override;
  };

} // namespace poseidon2

#endif