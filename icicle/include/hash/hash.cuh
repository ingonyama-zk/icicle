#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

namespace hash {
  template <typename Image>
  class Permutation
  {
  public:
    unsigned int width;
    unsigned int preimage_max_length = width;

    virtual cudaError_t squeeze_states(
      const Image* states,
      unsigned int number_of_states,
      unsigned int rate,
      unsigned int offset,
      Image* output,
      device_context::DeviceContext& ctx) const = 0;

    virtual cudaError_t run_permutation_kernel(
      const Image* states, Image* output, unsigned int number_of_states, device_context::DeviceContext& ctx) const = 0;

    cudaError_t permute_many(
      const Image* states,
      Image* output,
      unsigned int number_of_states,
      device_context::DeviceContext& ctx,
      bool is_async)
    {
      CHK_INIT_IF_RETURN();

      bool states_on_device = !device_context::is_host_ptr(states, ctx.device_id);
      bool output_on_device = !device_context::is_host_ptr(output, ctx.device_id);

      if (states_on_device != output_on_device)
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "States and output should be both on-device or on-host");

      S* d_states = nullptr;
      if (!states_on_device) {
        CHK_IF_RETURN(cudaMallocAsync(&d_states, number_of_states * this->width * sizeof(Image), ctx.stream));
        CHK_IF_RETURN(cudaMemcpyAsync(
          d_states, states, number_of_states * this->width * sizeof(Image), cudaMemcpyHostToDevice, ctx.stream));
      }

      cudaError_t permutation_error = this->run_permutation_kernel(
        states_on_device ? states : d_states, output_on_device ? output : d_states, number_of_states, ctx);
      CHK_IF_RETURN(permutation_error);

      if (!states_on_device) {
        CHK_IF_RETURN(cudaMemcpyAsync(
          output, d_states, number_of_states * this->width * sizeof(Image), cudaMemcpyDeviceToHost, ctx.stream));
        CHK_IF_RETURN(cudaFreeAsync(d_states, ctx.stream))
      }

      if (!is_async) CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      return CHK_LAST();
    }
  };

  template <typename Image>
  class CompressionHasher : Permutation<Image>
  {
  public:
    virtual cudaError_t compress_many(
      const Image* states,
      Image* output,
      unsigned int number_of_states,
      unsigned int offset,
      Image* perm_output,
      device_context::DeviceContext& ctx,
      bool is_async) const
    {
      CHK_INIT_IF_RETURN();
      bool states_on_device = !device_context::is_host_ptr(states, ctx.device_id);
      bool output_on_device = !device_context::is_host_ptr(output, ctx.device_id);

      // Allocate memory for states if needed
      bool need_allocate_perm_output = perm_output == nullptr;
      if (need_allocate_perm_output) {
        CHK_IF_RETURN(cudaMallocAsync(&perm_output, number_of_states * this->width * sizeof(Image), ctx.stream))
      }

      // Copy states from host if they are on host
      if (!states_on_device) {
        CHK_IF_RETURN(cudaMemcpyAsync(
          perm_output, states, number_of_states * this->width * sizeof(Image), cudaMemcpyHostToDevice, ctx.stream));
      }

      // Run the permutation
      cudaError_t hash_error =
        this->run_permutation_kernel(states_on_device ? states : perm_output, perm_output, number_of_states, ctx);
      CHK_IF_RETURN(hash_error);

      // Squeeze states to get results
      cudaError_t squeeze_error = this->squeeze_states(perm_output, number_of_states, 1, perm_output, ctx, offset);
      CHK_IF_RETURN(squeeze_error);

      if (output != perm_output) {
        CHK_IF_RETURN(cudaMemcpyAsync(
          output, perm_output, number_of_states * sizeof(Image),
          output_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, ctx.stream));
      }

      if (need_allocate_perm_output) CHK_IF_RETURN(cudaFreeAsync(perm_output, ctx.stream))

      if (!is_async) CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      return CHK_LAST();
    }
  };

  struct SpongeConfig {
    bool are_inputs_on_device;
    bool are_outputs_on_device;
    unsigned int input_block_len;
    unsigned int output_len;
    unsigned int input_rate;
    unsigned int output_rate;
    unsigned int offset;
    device_context::DeviceContext ctx;
    bool is_async;
  };

  template <typename PreImage, typename Image>
  class SpongeHasher : Permutation<Image>
  {
    virtual cudaError_t pad_many(Image* states, unsigned int number_of_states, SpongeConfig& cfg) const = 0;

  public:
    cudaError_t absorb_many(const PreImage* inputs, Image* states, unsigned int number_of_states, SpongeConfig& cfg)
    {
      if (cfg.input_block_len > cfg.rate)
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Input blocks can not be bigger than hash rate");
      // This allows to copy hash inputs and apply zero padding
      // If width and preimage_max_len is the same
      CHK_IF_RETURN(cudaMemcpy2DAsync(
        states, this->width * sizeof(Image),    // (Dst) States pointer and pitch
        inputs, this->rate * sizeof(PreImage),  // (Src) Inputs pointer and pitch
        cfg.input_block_len * sizeof(PreImage), // Width of the source matrix
        number_of_states,                       // Height of the source matrix
        cfg.are_inputs_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, cfg.ctx.stream));

      cudaError_t padding_error = this->pad_many(states, number_of_states, );
      CHK_IF_RETURN(padding_error);

      cudaError_t permutation_error = this->permute_many(states, states, number_of_states, cfg.ctx, cfg.is_async);
      CHK_IF_RETURN(permutation_error);

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      return CHK_LAST();
    }

    cudaError_t squeeze_many(Image* states, Image* output, unsigned int number_of_states, SpongeConfig& cfg)
    {
      Image* d_output;
      if (!cfg.are_outputs_on_device) {
        CHK_IF_RETURN(cudaMallocAsync(&d_output, number_of_states * cfg.output_len * sizeof(Image), cfg.ctx.stream))
      } else {
        d_output = output;
      }

      cudaError_t squeeze_error = this->squeeze_states(states, number_of_states, cfg.rate, cfg.offset, output, cfg.ctx);
      CHK_IF_RETURN(squeeze_error);

      if (!cfg.are_outputs_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_output, cfg.ctx.stream)); }

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      return CHK_LAST();
    }

    cudaError_t hash_many(const PreImage* inputs, Image* output, unsigned int number_of_states, SpongeConfig& cfg)
    {
      CHK_INIT_IF_RETURN();

      Image* states;
      CHK_IF_RETURN(cudaMallocAsync(&states, number_of_states * this->width * sizeof(Image), cfg.ctx.stream))

      CHK_IF_RETURN(absorb_many(inputs, states, number_of_states, cfg));
      CHK_IF_RETURN(squeeze_many(states, output, number_of_states, cfg));

      CHK_IF_RETURN(cudaFreeAsync(states, cfg.ctx.stream));
      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      return CHK_LAST();
    }
  };
} // namespace hash

#endif