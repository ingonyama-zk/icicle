#pragma once
#ifndef POSEIDON2_H
#define POSEIDON2_H

#include <cstdint>
#include <stdexcept>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"

#include "hash/hash.cuh"
#include "matrix/matrix.cuh"

#include "poseidon2/constants.cuh"
#include "poseidon2/kernels.cuh"

using matrix::Matrix;

/**
 * @namespace poseidon2
 * Implementation of the [Poseidon2 hash function](https://eprint.iacr.org/2019/458.pdf)
 * Specifically, the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/)
 */
namespace poseidon2 {
  template <typename S>
  class Poseidon2 : public hash::Hasher<S, S>
  {
    static const int POSEIDON_BLOCK_SIZE = 32;

    static inline int poseidon_number_of_blocks(size_t number_of_states)
    {
      return number_of_states / POSEIDON_BLOCK_SIZE + static_cast<bool>(number_of_states % POSEIDON_BLOCK_SIZE);
    }

  public:
    const std::size_t device_id;
    Poseidon2Constants<S> constants;

    cudaError_t hash_2d(
      const Matrix<S>* inputs,
      S* output,
      unsigned int number_of_inputs,
      unsigned int output_len,
      uint64_t number_of_rows,
      const device_context::DeviceContext& ctx) const override
    {
#define P2_HASH_2D_T(width)                                                                                            \
  case width:                                                                                                          \
    hash_2d_kernel<S, width><<<poseidon_number_of_blocks(number_of_rows), POSEIDON_BLOCK_SIZE, 0, ctx.stream>>>(       \
      inputs, output, number_of_inputs, this->rate, output_len, this->constants);                                      \
    break;

      switch (this->width) {
        P2_HASH_2D_T(2)
        P2_HASH_2D_T(3)
        P2_HASH_2D_T(4)
        P2_HASH_2D_T(8)
        P2_HASH_2D_T(12)
        P2_HASH_2D_T(16)
        P2_HASH_2D_T(20)
        P2_HASH_2D_T(24)
      default:
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "PoseidonAbsorb2d: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
      }

      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    cudaError_t run_hash_many_kernel(
      const S* input,
      S* output,
      unsigned int number_of_states,
      unsigned int input_len,
      unsigned int output_len,
      const device_context::DeviceContext& ctx) const override
    {
#define P2_HASH_MANY_T(width)                                                                                          \
  case width:                                                                                                          \
    hash_many_kernel<S, width><<<poseidon_number_of_blocks(number_of_states), POSEIDON_BLOCK_SIZE, 0, ctx.stream>>>(   \
      input, output, number_of_states, input_len, output_len, this->constants);                                        \
    break;

      switch (this->width) {
        P2_HASH_MANY_T(2)
        P2_HASH_MANY_T(3)
        P2_HASH_MANY_T(4)
        P2_HASH_MANY_T(8)
        P2_HASH_MANY_T(12)
        P2_HASH_MANY_T(16)
        P2_HASH_MANY_T(20)
        P2_HASH_MANY_T(24)
      default:
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "PoseidonPermutation: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
      }
      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    cudaError_t compress_and_inject(
      const Matrix<S>* matrices_to_inject,
      unsigned int number_of_inputs,
      uint64_t number_of_rows,
      const S* prev_layer,
      S* next_layer,
      unsigned int digest_elements,
      const device_context::DeviceContext& ctx) const override
    {
#define P2_COMPRESS_AND_INJECT_T(width)                                                                                \
  case width:                                                                                                          \
    compress_and_inject_kernel<S, width>                                                                               \
      <<<poseidon_number_of_blocks(number_of_rows), POSEIDON_BLOCK_SIZE, 0, ctx.stream>>>(                             \
        matrices_to_inject, number_of_inputs, prev_layer, next_layer, this->rate, digest_elements, this->constants);   \
    break;

      switch (this->width) {
        P2_COMPRESS_AND_INJECT_T(2)
        P2_COMPRESS_AND_INJECT_T(3)
        P2_COMPRESS_AND_INJECT_T(4)
        P2_COMPRESS_AND_INJECT_T(8)
        P2_COMPRESS_AND_INJECT_T(12)
        P2_COMPRESS_AND_INJECT_T(16)
        P2_COMPRESS_AND_INJECT_T(20)
        P2_COMPRESS_AND_INJECT_T(24)
      default:
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "PoseidonPermutation: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
      }

      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    Poseidon2(
      unsigned int width,
      unsigned int rate,
      unsigned int alpha,
      unsigned int internal_rounds,
      unsigned int external_rounds,
      const S* round_constants,
      const S* internal_matrix_diag,
      MdsType mds_type,
      DiffusionStrategy diffusion,
      device_context::DeviceContext& ctx)
        : hash::Hasher<S, S>(width, width, rate, 0), device_id(ctx.device_id)
    {
      Poseidon2Constants<S> constants;
      CHK_STICKY(create_poseidon2_constants(
        width, alpha, internal_rounds, external_rounds, round_constants, internal_matrix_diag, mds_type, diffusion, ctx,
        &constants));
      this->constants = constants;
    }

    Poseidon2(
      unsigned int width,
      unsigned int rate,
      MdsType mds_type,
      DiffusionStrategy diffusion,
      device_context::DeviceContext& ctx)
        : hash::Hasher<S, S>(width, width, rate, 0), device_id(ctx.device_id)
    {
      Poseidon2Constants<S> constants;
      CHK_STICKY(init_poseidon2_constants(width, mds_type, diffusion, ctx, &constants));
      this->constants = constants;
    }

    ~Poseidon2()
    {
      auto ctx = device_context::get_default_device_context();
      ctx.device_id = this->device_id;
      CHK_STICKY(release_poseidon2_constants<S>(&this->constants, ctx));
    }
  };

} // namespace poseidon2

#endif