#pragma once
#ifndef POSEIDON2_H
#define POSEIDON2_H

#include <cstdint>
#include <stdexcept>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"
#include "hash/hash.cuh"

#include "constants.cuh"
#include "kernels.cuh"

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
  cudaError_t release_poseidon2_constants(Poseidon2Constants<S>* constants, device_context::DeviceContext& ctx);

  template <typename S>
  class Poseidon2 : public Permutation<S>, public CompressionHasher<S>, public SpongeHasher<S, S>
  {
    static const int POSEIDON_BLOCK_SIZE = 128;

    static inline int poseidon_number_of_blocks(size_t number_of_states)
    {
      return number_of_states / POSEIDON_BLOCK_SIZE + static_cast<bool>(number_of_states % POSEIDON_BLOCK_SIZE);
    }

    Poseidon2Constants<S> constants;

    cudaError_t squeeze_states(
      const S* states,
      unsigned int number_of_states,
      unsigned int rate,
      S* output,
      device_context::DeviceContext& ctx,
      unsigned int offset = 0) const override
    {
#define P2_SQUEEZE_T(width)                                                                                            \
  case width:                                                                                                          \
    squeeze_states_kernel<S, width, 1, 0>                                                                              \
      <<<poseidon_number_of_blocks(number_of_states), POSEIDON_BLOCK_SIZE, 0, ctx.stream>>>(                           \
        states, number_of_states, output);                                                                             \
    break;

      switch (this->width) {
        P2_SQUEEZE_T(2)
        P2_SQUEEZE_T(3)
        P2_SQUEEZE_T(4)
        P2_SQUEEZE_T(8)
        P2_SQUEEZE_T(12)
        P2_SQUEEZE_T(16)
        P2_SQUEEZE_T(20)
        P2_SQUEEZE_T(24)
      default:
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "PoseidonSqueeze: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
      }
      // Squeeze states to get results
      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    cudaError_t run_permutation_kernel(
      const S* states, S* output, unsigned int number_of_states, device_context::DeviceContext& ctx) const override
    {
#define P2_PERM_T(width)                                                                                               \
  case width:                                                                                                          \
    poseidon2_permutation_kernel<S, width>                                                                             \
      <<<poseidon_number_of_blocks(number_of_states), POSEIDON_BLOCK_SIZE, 0, ctx.stream>>>(                           \
        states, output, number_of_states, this->constants);                                                            \
    break;

      switch (this->width) {
        P2_PERM_T(2)
        P2_PERM_T(3)
        P2_PERM_T(4)
        P2_PERM_T(8)
        P2_PERM_T(12)
        P2_PERM_T(16)
        P2_PERM_T(20)
        P2_PERM_T(24)
      default:
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "PoseidonPermutation: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
      }

      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

  public:
    Poseidon2(
      unsigned int width,
      unsigned int alpha,
      unsigned int internal_rounds,
      unsigned int external_rounds,
      const S* round_constants,
      const S* internal_matrix_diag,
      MdsType mds_type,
      DiffusionStrategy diffusion,
      device_context::DeviceContext& ctx)
    {
      Poseidon2Constants<S> constants;
      CHK_STICKY(create_poseidon2_constants(
        width, alpha, internal_rounds, external_rounds, round_constants, internal_matrix_diag, mds_type, diffusion, ctx,
        &constants));
      this->constants = constants;
      this->width = width;
    }

    Poseidon2(int width, MdsType mds_type, DiffusionStrategy diffusion, device_context::DeviceContext& ctx)
    {
      Poseidon2Constants<S> constants;
      CHK_STICKY(init_poseidon2_constants(width, mds_type, diffusion, ctx, &constants));
      this->constants = constants;
      this->width = width;
    }

    ~Poseidon2()
    {
      auto ctx = device_context::get_default_device_context();
      CHK_STICKY(release_poseidon2_constants<S>(&this->constants, ctx));
    }
  };

} // namespace poseidon2

#endif