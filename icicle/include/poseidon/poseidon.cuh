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
  /**
   * @struct PoseidonConfig
   * Struct that encodes various Poseidon parameters.
   */
  struct PoseidonConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool input_is_a_state;      /**< If true, input is considered to be a states vector, holding the preimages
                                 * in aligned or not aligned format. Memory under the input pointer will be used for states
                                 * If false, fresh states memory will be allocated and input will be copied into it */
    bool aligned;               /**< If true - input should be already aligned for poseidon permutation.
                                 * Aligned format: [0, A, B, 0, C, D, ...] (as you might get by using loop_state)
                                 * not aligned format: [A, B, 0, C, D, 0, ...] (as you might get from cudaMemcpy2D) */
    bool loop_state;            /**< If true, hash results will also be copied in the input pointer in aligned format */
    bool is_async; /**< Whether to run the Poseidon asynchronously. If set to `true`, the poseidon_hash function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the poseidon_hash
                    *   function will block the current CPU thread. */
  };

  static PoseidonConfig default_poseidon_config(
    int t, const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    PoseidonConfig config = {
      ctx,   // ctx
      false, // are_inputes_on_device
      false, // are_outputs_on_device
      false, // input_is_a_state
      false, // aligned
      false, // loop_state
      false, // is_async
    };
    return config;
  }

  static SpongeConfig default_poseidon_sponge_config(
    int width, const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    SpongeConfig cfg = default_sponge_config(ctx);
    cfg.input_rate = width - 1;
    cfg.output_rate = cfg.input_rate;
    cfg.offset = 1;
    return cfg;
  }

  /**
   * Compute the poseidon hash over a sequence of preimages.
   * Takes {number_of_states * (T-1)} elements of input and computes {number_of_states} hash images
   * @param T size of the poseidon state, should be equal to {arity + 1}
   * @param input a pointer to the input data. May be allocated on device or on host, regulated
   * by the config. May point to a string of preimages or a string of states filled with preimages.
   * @param output a pointer to the output data. May be allocated on device or on host, regulated
   * by the config. Must be at least of size [number_of_states](@ref number_of_states)
   * @param number_of_states number of input blocks of size T-1 (arity)
   */
  template <typename S, int T>
  cudaError_t poseidon_hash(
    S* input, S* output, size_t number_of_states, const PoseidonConstants<S>& constants, const PoseidonConfig& config);

  template <typename S>
  class Poseidon : public Hash<S>, public SpongeHasher<Poseidon<S>, S, S>, public CompressionHasher<Poseidon<S>, S>
  {
  public:
    PoseidonConstants<S> constants;

    cudaError_t squeeze_states(
      const S* states,
      unsigned int number_of_states,
      unsigned int rate,
      unsigned int offset,
      bool align,
      S* output,
      const device_context::DeviceContext& ctx) const override
    {
      if (align && rate == 1) {
        squeeze_states_kernel<S>
          <<<PKC::number_of_singlehash_blocks(number_of_states), PKC::singlehash_block_size, 0, ctx.stream>>>(
            states, number_of_states, this->width, output);
      } else {
        generic_squeeze_states_kernel<S>
          <<<PKC::number_of_singlehash_blocks(number_of_states), PKC::singlehash_block_size, 0, ctx.stream>>>(
            states, number_of_states, this->width, rate, offset, output);
      }
      // Squeeze states to get results
      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    cudaError_t run_permutation_kernel(
      const S* states, S* output, unsigned int number_of_states, bool aligned, const device_context::DeviceContext& ctx) const override
    {
      cudaError_t permutation_error;
#define P_PERM_T(width)                                                                                                \
  case width:                                                                                                          \
    permutation_error =                                                                                                \
      poseidon_permutation_kernel<S, width>(states, output, number_of_states, aligned, this->constants, ctx.stream);            \
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
    {
      PoseidonConstants<S> constants;
      CHK_STICKY(create_optimized_poseidon_constants(
        arity, alpha, partial_rounds, full_rounds_half, round_constants, mds_matrix, non_sparse_matrix, sparse_matrices,
        domain_tag, &constants, ctx));
      this->constants = constants;
      this->preimage_max_length = arity;
      this->width = arity + 1;
    }

    Poseidon(int arity, device_context::DeviceContext& ctx)
    {
      PoseidonConstants<S> constants{};
      CHK_STICKY(init_optimized_poseidon_constants(arity, ctx, &constants));
      this->constants = constants;
      this->preimage_max_length = arity;
      this->width = arity + 1;
    }

    ~Poseidon()
    {
      auto ctx = device_context::get_default_device_context();
      CHK_STICKY(release_optimized_poseidon_constants<S>(&this->constants, ctx));
    }
  };
} // namespace poseidon

#endif