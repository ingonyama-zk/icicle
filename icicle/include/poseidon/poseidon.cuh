#pragma once
#ifndef POSEIDON_H
#define POSEIDON_H

#include <cstdint>
#include <stdexcept>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"

#include "poseidon/kernels.cuh"
#include "poseidon/constants.cuh"
#include "hash/hash.cuh"
using namespace hash;

/**
 * @namespace poseidon
 * Implementation of the [Poseidon hash function](https://eprint.iacr.org/2019/458.pdf)
 * Specifically, the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/)
 */
namespace poseidon {
  template <typename S>
  class Poseidon : public SpongeHasher<S, S>
  {
  public:
    const std::size_t device_id;
    PoseidonConstants<S> constants;

    cudaError_t run_hash_many_kernel(
      const S* input,
      S* output,
      unsigned int number_of_states,
      unsigned int input_len,
      unsigned int output_len,
      const device_context::DeviceContext& ctx) const override
    {
      cudaError_t permutation_error;
#define P_PERM_T(width)                                                                                                \
  case width:                                                                                                          \
    permutation_error = poseidon_permutation_kernel<S, width>(                                                         \
      input, output, number_of_states, input_len, output_len, this->constants, ctx.stream);                            \
    break;

      switch (this->width) {
        P_PERM_T(3)
        P_PERM_T(5)
        P_PERM_T(9)
        P_PERM_T(12)
      default:
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "PoseidonPermutation: #width must be one of [3, 5, 9, 12]");
      }

      CHK_IF_RETURN(permutation_error);
      return CHK_LAST();
    }

    Poseidon(
      unsigned int arity,
      unsigned int alpha,
      unsigned int partial_rounds,
      unsigned int full_rounds_half,
      const S* round_constants,
      const S* mds_matrix,
      const S* non_sparse_matrix,
      const S* sparse_matrices,
      const S domain_tag,
      device_context::DeviceContext& ctx)
        : SpongeHasher<S, S>(arity + 1, arity, arity, 1), device_id(ctx.device_id)
    {
      PoseidonConstants<S> constants;
      CHK_STICKY(create_optimized_poseidon_constants(
        arity, alpha, partial_rounds, full_rounds_half, round_constants, mds_matrix, non_sparse_matrix, sparse_matrices,
        domain_tag, &constants, ctx));
      this->constants = constants;
    }

    Poseidon(int arity, device_context::DeviceContext& ctx)
        : SpongeHasher<S, S>(arity + 1, arity, arity, 1), device_id(ctx.device_id)
    {
      PoseidonConstants<S> constants{};
      CHK_STICKY(init_optimized_poseidon_constants(arity, ctx, &constants));
      this->constants = constants;
    }

    ~Poseidon()
    {
      auto ctx = device_context::get_default_device_context();
      ctx.device_id = this->device_id;
      CHK_STICKY(release_optimized_poseidon_constants<S>(&this->constants, ctx));
    }
  };
} // namespace poseidon

#endif