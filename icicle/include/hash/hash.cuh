#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include <cassert>

namespace hash {
  struct SpongeConfig {
    device_context::DeviceContext ctx;
    bool are_inputs_on_device;
    bool are_outputs_on_device;
    bool is_async;
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
   * @param offset Squeeze offset. Start squeezing from Oth element of the state
   * @param out pointer for squeeze results. Can be equal to states to do in-place squeeze
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
      out[idx * rate + i] = states[idx * width + offset];
    }
  }

  template <typename PreImage, typename Image>
  class SpongeHasher
  {
  public:
    const unsigned int width;
    const unsigned int preimage_max_length;
    const unsigned int rate;
    const unsigned int offset;

    SpongeHasher(unsigned int width, unsigned int preimage_max_length, unsigned int rate, unsigned int offset)
        : width(width), preimage_max_length(preimage_max_length), rate(rate), offset(offset)
    {
      assert(
        rate * sizeof(PreImage) <= preimage_max_length * sizeof(Image) &&
        "Input rate can not be bigger than preimage max length");
    }

    virtual cudaError_t pad_many(
      Image* states,
      unsigned int number_of_states,
      unsigned int input_block_len,
      const device_context::DeviceContext& ctx) const
    {
      return cudaError_t::cudaSuccess;
    };

    virtual cudaError_t squeeze_states(
      const Image* states,
      unsigned int number_of_states,
      unsigned int output_len,
      Image* output,
      const device_context::DeviceContext& ctx) const = 0;

    virtual cudaError_t run_permutation_kernel(
      const Image* states,
      Image* output,
      unsigned int number_of_states,
      bool aligned,
      const device_context::DeviceContext& ctx) const = 0;

    virtual cudaError_t prepare_states(
      const Image* input, Image* out, unsigned int number_of_states, const device_context::DeviceContext& ctx) const
    {
      return cudaError_t::cudaSuccess;
    };

    cudaError_t compress_many(
      Image* input, Image* out, unsigned int number_of_states, const device_context::DeviceContext ctx) const
    {
      CHK_IF_RETURN(run_permutation_kernel(input, input, number_of_states, true, ctx));
      CHK_IF_RETURN(squeeze_states(input, number_of_states, 1, out, ctx));

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

      std::cout << input_block_len << std::endl;
      std::cout << width << std::endl;
      std::cout << rate << std::endl;
      std::cout << number_of_states << std::endl;
      // This allows to copy hash inputs and apply zero padding
      CHK_IF_RETURN(cudaMemcpy2DAsync(
        states, width * sizeof(Image),      // (Dst) States pointer and pitch
        inputs, rate * sizeof(Image),       // (Src) Inputs pointer and pitch
        input_block_len * sizeof(PreImage), // Width of the source matrix
        number_of_states,                   // Height of the source matrix
        cfg.are_inputs_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, cfg.ctx.stream));

      CHK_IF_RETURN(pad_many(states, number_of_states, input_block_len, cfg.ctx));
      CHK_IF_RETURN(run_permutation_kernel(states, states, number_of_states, false, cfg.ctx));

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));

      return CHK_LAST();
    }

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