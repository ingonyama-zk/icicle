#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include <cassert>

/**
 * @namespace hash
 * Includes classes and methods for describing hash functions.
 */
namespace hash {

  /**
   * @struct SpongeConfig
   * Encodes sponge hash operations parameters.
   */
  struct SpongeConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    bool are_inputs_on_device; /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool
      are_outputs_on_device; /**< True if outputs are on device and false if they're on host. Default value: false. */
    bool is_async; /**< Whether to run the hash operations asynchronously. If set to `true`, the functions will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false,
                    *   functions will block the current CPU thread. */
  };

  /**
   * A function that returns the default value of [SpongeConfig](@ref SpongeConfig) for the [SpongeHasher](@ref
   * SpongeHasher) class.
   * @return Default value of [SpongeConfig](@ref SpongeConfig).
   */
  static SpongeConfig
  default_sponge_config(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    SpongeConfig config = {
      ctx,   // ctx
      false, // are_inputs_on_device
      false, // are_outputs_on_device
      false, // is_async
    };
    return config;
  }

  /**
   * Squeeze states to extract the results.
   * 1 GPU thread operates on 1 state.
   *
   * @param states the states to squeeze
   * @param number_of_states number of states to squeeze
   * @param width Width of the state
   * @param rate Squeeze rate. How many elements to extract from each state
   * @param offset Squeeze offset
   * @param out pointer for squeeze results. Warning: out can't be equal to states
   *
   * @tparam S Type of the state element
   */
  template <typename S>
  __global__ void generic_squeeze_states_kernel(
    const S* states, unsigned int number_of_states, unsigned int width, unsigned int rate, unsigned int offset, S* out)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    for (int i = 0; i < rate; i++) {
      out[idx * rate + i] = states[idx * width + offset + i];
    }
  }

  /**
   * @class SpongeHasher
   *
   * Can be inherited by a cryptographic permutation function to create a
   * [sponge](https://en.wikipedia.org/wiki/Sponge_function) construction out of it.
   *
   * @tparam PreImage type of inputs elements
   * @tparam Image type of state elements. Also used to describe the type of hash output
   */
  template <typename PreImage, typename Image>
  class SpongeHasher
  {
  public:
    /// @brief the width of permutation state
    const unsigned int width;

    /// @brief how many elements a state can fit per 1 permutation. Used with domain separation.
    const unsigned int preimage_max_length;

    /// @brief portion of the state to absorb input into, or squeeze output from
    const unsigned int rate;

    /// @brief start squeezing from this offset. Used with domain separation.
    const unsigned int offset;

    SpongeHasher(unsigned int width, unsigned int preimage_max_length, unsigned int rate, unsigned int offset)
        : width(width), preimage_max_length(preimage_max_length), rate(rate), offset(offset)
    {
      assert(
        rate * sizeof(PreImage) <= preimage_max_length * sizeof(Image) &&
        "Input rate can not be bigger than preimage max length");
    }

    /// @brief Used to pad input in absorb function.
    /// @param states pointer to states allocated on-device
    virtual cudaError_t pad_many(
      Image* states,
      unsigned int number_of_states,
      unsigned int input_block_len,
      const device_context::DeviceContext& ctx) const
    {
      return cudaError_t::cudaSuccess;
    };

    /// @brief Squeeze hash output from states
    /// @param states pointer to states allocated on-device
    /// @param output pointer to output allocated on-device
    virtual cudaError_t squeeze_states(
      const Image* states,
      unsigned int number_of_states,
      unsigned int output_len,
      Image* output,
      const device_context::DeviceContext& ctx) const = 0;

    /// @brief Run the cryptographic permutation function.
    /// @param states pointer to states allocated on-device
    /// @param output pointer to states allocated on-device. Can equal to @ref(states) to run in-place
    /// @param aligned if set to true, some permutation can skip the alignment step. E.G. poseidon1
    virtual cudaError_t run_permutation_kernel(
      const Image* states,
      Image* output,
      unsigned int number_of_states,
      bool aligned,
      const device_context::DeviceContext& ctx) const = 0;

    /// @brief Aligns states. Used with domain separation
    /// @param input pointer to input allocated on-device
    /// @param out pointer to memory to write the aligned states
    /// @param number_of_states equals to the number of elements inside input
    virtual cudaError_t prepare_states(
      const Image* input, Image* out, unsigned int number_of_states, const device_context::DeviceContext& ctx) const
    {
      return cudaError_t::cudaSuccess;
    };

    /// @brief Permute aligned input and do squeeze
    /// @param input pointer to input allocated on-device
    /// @param out pointer to output allocated on-device
    cudaError_t compress_many(
      Image* input,
      Image* out,
      unsigned int number_of_states,
      unsigned int output_len,
      const device_context::DeviceContext ctx) const
    {
      CHK_IF_RETURN(run_permutation_kernel(input, input, number_of_states, true, ctx));
      CHK_IF_RETURN(squeeze_states(input, number_of_states, output_len, out, ctx));

      return CHK_LAST();
    }

    cudaError_t
    allocate_states(Image** states, unsigned int number_of_states, const device_context::DeviceContext& ctx) const
    {
      CHK_INIT_IF_RETURN();
      return cudaMallocAsync(states, number_of_states * width * sizeof(Image), ctx.stream);
    }

    cudaError_t free_states(Image* states, const device_context::DeviceContext& ctx) const
    {
      CHK_INIT_IF_RETURN();
      return cudaFreeAsync(states, ctx.stream);
    }

    /// @brief Copy inputs to states, aligning them if needed. Run the permutation on states
    /// @param inputs pointer to inputs on-device or on-host
    /// @param states pointer to states allocated on-device
    /// @param input_block_len number of input elements for each state
    cudaError_t absorb_many(
      const PreImage* inputs,
      Image* states,
      unsigned int number_of_states,
      unsigned int input_block_len,
      const SpongeConfig& cfg) const
    {
      CHK_INIT_IF_RETURN();

      if (input_block_len * sizeof(PreImage) > rate * sizeof(Image))
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Input blocks can not be bigger than hash rate");

      // This allows to copy hash inputs and apply zero padding
      CHK_IF_RETURN(cudaMemcpy2DAsync(
        states, width * sizeof(Image),           // (Dst) States pointer and pitch
        inputs, input_block_len * sizeof(Image), // (Src) Inputs pointer and pitch
        input_block_len * sizeof(PreImage),      // Width of the source matrix
        number_of_states,                        // Height of the source matrix
        cfg.are_inputs_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, cfg.ctx.stream));

      CHK_IF_RETURN(pad_many(states, number_of_states, input_block_len, cfg.ctx));
      CHK_IF_RETURN(run_permutation_kernel(states, states, number_of_states, false, cfg.ctx));

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));

      return CHK_LAST();
    }

    /// @brief extract elements from states
    /// @param states should be on-device
    /// @param output can be on-host or on-device
    /// @param output_len number of elements to squeeze per state
    cudaError_t squeeze_many(
      const Image* states,
      Image* output,
      unsigned int number_of_states,
      unsigned int output_len,
      const SpongeConfig& cfg) const
    {
      if (rate < output_len)
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Output len can not be bigger than output rate");

      Image* d_output;
      if (!cfg.are_outputs_on_device) {
        CHK_IF_RETURN(cudaMallocAsync(&d_output, number_of_states * output_len * sizeof(Image), cfg.ctx.stream))
      } else {
        d_output = output;
      }

      CHK_IF_RETURN(squeeze_states(states, number_of_states, output_len, d_output, cfg.ctx));

      if (!cfg.are_outputs_on_device) {
        CHK_IF_RETURN(cudaMemcpyAsync(
          output, d_output, number_of_states * output_len * sizeof(Image), cudaMemcpyDeviceToHost, cfg.ctx.stream));
        CHK_IF_RETURN(cudaFreeAsync(d_output, cfg.ctx.stream));
      }

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));

      return CHK_LAST();
    }

    /// @brief Allocates states on-device and does absorb + squeeze
    /// @param inputs can be on-host or on-device
    /// @param output can be on-host or on-device
    /// @param input_block_len number of elements per state
    /// @param output_len number of output elements per state
    cudaError_t hash_many(
      const PreImage* inputs,
      Image* output,
      unsigned int number_of_states,
      unsigned int input_block_len,
      unsigned int output_len,
      const SpongeConfig& cfg) const
    {
      CHK_INIT_IF_RETURN();

      Image* states;
      CHK_IF_RETURN(allocate_states(&states, number_of_states, cfg.ctx));

      CHK_IF_RETURN(absorb_many(inputs, states, number_of_states, input_block_len, cfg));
      CHK_IF_RETURN(squeeze_many(states, output, number_of_states, output_len, cfg));

      CHK_IF_RETURN(free_states(states, cfg.ctx));
      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));

      return CHK_LAST();
    }
  };
} // namespace hash

#endif