#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

namespace hash {
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
      if (states == out) {
        S element = states[idx * width + offset];
        __syncthreads();
        out[idx * rate + i] = element;
      } else {
        out[idx * rate + i] = states[idx * width + offset];
      }
    }
  }

  template <typename Image>
  class Hash
  {
  public:
    unsigned int width;
    unsigned int preimage_max_length;

    Hash() = default;
    Hash(unsigned int width, unsigned int preimage_max_length) : width(width), preimage_max_length(preimage_max_length)
    {
    }

    virtual cudaError_t pad_many(
      Image* states,
      unsigned int number_of_states,
      unsigned int input_block_len,
      unsigned int rate,
      device_context::DeviceContext& ctx) const
    {
      return cudaError_t::cudaSuccess;
    };

    virtual cudaError_t squeeze_states(
      const Image* states,
      unsigned int number_of_states,
      unsigned int rate,
      unsigned int offset,
      Image* output,
      device_context::DeviceContext& ctx) const = 0;

    virtual cudaError_t run_permutation_kernel(
      const Image* states, Image* output, unsigned int number_of_states, device_context::DeviceContext& ctx) const = 0;
  };

  struct SpongeConfig {
    device_context::DeviceContext ctx;
    bool are_inputs_on_device;
    bool are_outputs_on_device;
    unsigned int input_rate;
    unsigned int output_rate;
    unsigned int offset;
    bool recursive_squeeze;
    bool aligned;
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
      0,     // input_rate
      0,     // output_rate
      0,     // offset
      false, // recursive_squeeze
      false, // recursive_squeeze
      false, // is_async
    };
    return config;
  }

  template <typename H, typename PreImage, typename Image>
  class SpongeHasher
  {
  public:
    cudaError_t allocate_states(Image** states, unsigned int number_of_states, device_context::DeviceContext& ctx) const
    {
      CHK_INIT_IF_RETURN();
      return cudaMallocAsync(states, number_of_states * static_cast<const H*>(this)->width * sizeof(Image), ctx.stream);
    }

    cudaError_t free_states(Image* states, device_context::DeviceContext& ctx) const
    {
      CHK_INIT_IF_RETURN();
      return cudaFreeAsync(states, ctx.stream);
    }

    cudaError_t absorb_many(
      const PreImage* inputs,
      Image* states,
      unsigned int number_of_states,
      unsigned int input_block_len,
      SpongeConfig& cfg) const
    {
      CHK_INIT_IF_RETURN();
      unsigned int width = static_cast<const H*>(this)->width;
      unsigned int preimage_max_length = static_cast<const H*>(this)->preimage_max_length;

      if (cfg.input_rate * sizeof(PreImage) > preimage_max_length * sizeof(Image))
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Input rate can not be bigger than preimage max length");
      if (input_block_len > cfg.input_rate)
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Input blocks can not be bigger than hash rate");

      // This allows to copy hash inputs and apply zero padding
      CHK_IF_RETURN(cudaMemcpy2DAsync(
        states, width * sizeof(Image),             // (Dst) States pointer and pitch
        inputs, cfg.input_rate * sizeof(PreImage), // (Src) Inputs pointer and pitch
        input_block_len * sizeof(PreImage),        // Width of the source matrix
        number_of_states,                          // Height of the source matrix
        cfg.are_inputs_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, cfg.ctx.stream));

      cudaError_t padding_error =
        static_cast<const H*>(this)->pad_many(states, number_of_states, input_block_len, cfg.input_rate, cfg.ctx);
      CHK_IF_RETURN(padding_error);

      cudaError_t permutation_error =
        static_cast<const H*>(this)->run_permutation_kernel(states, states, number_of_states, cfg.ctx);
      CHK_IF_RETURN(permutation_error);

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));

      return CHK_LAST();
    }

    cudaError_t squeeze_many(
      const Image* states,
      Image* output,
      unsigned int number_of_states,
      unsigned int output_len,
      SpongeConfig& cfg) const
    {
      if (cfg.output_rate < output_len)
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Output len can not be bigger than output rate");

      Image* d_output;
      if (!cfg.are_outputs_on_device) {
        CHK_IF_RETURN(cudaMallocAsync(&d_output, number_of_states * output_len * sizeof(Image), cfg.ctx.stream))
      } else {
        d_output = output;
      }

      cudaError_t squeeze_error = static_cast<const H*>(this)->squeeze_states(
        states, number_of_states, output_len, cfg.offset, d_output, cfg.ctx);
      CHK_IF_RETURN(squeeze_error);

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
      SpongeConfig& cfg) const
    {
      CHK_INIT_IF_RETURN();
      unsigned int width = static_cast<const H*>(this)->width;

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