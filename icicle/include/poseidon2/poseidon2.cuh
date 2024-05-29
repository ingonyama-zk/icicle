#pragma once
#ifndef POSEIDON2_H
#define POSEIDON2_H

#include <cstdint>
#include <stdexcept>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"

#include "hash/hash.cuh"
using namespace hash;

#include "poseidon2/constants.cuh"
#include "poseidon2/kernels.cuh"

/**
 * @namespace poseidon2
 * Implementation of the [Poseidon2 hash function](https://eprint.iacr.org/2019/458.pdf)
 * Specifically, the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/)
 */
namespace poseidon2 {
  static SpongeConfig default_poseidon2_sponge_config(
    int width, const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    SpongeConfig cfg = default_sponge_config(ctx);
    cfg.input_rate = width;
    cfg.output_rate = cfg.input_rate;
    cfg.offset = 0;
    return cfg;
  }

  template <typename S>
  class Poseidon2 : public Hash<S>, public SpongeHasher<Poseidon2<S>, S, S>, public CompressionHasher<Poseidon2<S>, S>
  {
    static const int POSEIDON_BLOCK_SIZE = 128;

    static inline int poseidon_number_of_blocks(size_t number_of_states)
    {
      return number_of_states / POSEIDON_BLOCK_SIZE + static_cast<bool>(number_of_states % POSEIDON_BLOCK_SIZE);
    }

  public:
    Poseidon2Constants<S> constants;

    cudaError_t squeeze_states(
      const S* states,
      unsigned int number_of_states,
      unsigned int rate,
      unsigned int offset,
      bool align,
      S* output,
      const device_context::DeviceContext& ctx) const override
    {
      generic_squeeze_states_kernel<S>
        <<<poseidon_number_of_blocks(number_of_states), POSEIDON_BLOCK_SIZE, 0, ctx.stream>>>(
          states, number_of_states, this->width, rate, offset, output);
      // Squeeze states to get results
      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    cudaError_t run_permutation_kernel(
      const S* states, S* output, unsigned int number_of_states, bool aligned, device_context::DeviceContext& ctx) const override
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
      this->preimage_max_length = width;
    }

    Poseidon2(int width, MdsType mds_type, DiffusionStrategy diffusion, device_context::DeviceContext& ctx)
    {
      Poseidon2Constants<S> constants;
      CHK_STICKY(init_poseidon2_constants(width, mds_type, diffusion, ctx, &constants));
      this->constants = constants;
      this->width = width;
      this->preimage_max_length = width;
    }

    ~Poseidon2()
    {
      auto ctx = device_context::get_default_device_context();
      CHK_STICKY(release_poseidon2_constants<S>(&this->constants, ctx));
    }
  };

} // namespace poseidon2

#endif