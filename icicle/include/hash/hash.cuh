#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "matrix/matrix.cuh"
#include <cassert>

using matrix::Matrix;

/**
 * @namespace hash
 * Includes classes and methods for describing hash functions.
 */
namespace hash {

  /**
   * @struct HashConfig
   * Encodes hash operations parameters.
   */
  struct HashConfig {
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
   * A function that returns the default value of [HashConfig](@ref HashConfig) for the [Hasher](@ref
   * Hasher) class.
   * @return Default value of [HashConfig](@ref HashConfig).
   */
  static HashConfig
  default_hash_config(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    HashConfig config = {
      ctx,   // ctx
      false, // are_inputs_on_device
      false, // are_outputs_on_device
      false, // is_async
    };
    return config;
  }

  /**
   * @class Hasher
   *
   * An interface containing methods for hashing
   *
   * @tparam PreImage type of inputs elements
   * @tparam Image type of state elements. Also used to describe the type of hash output
   */
  template <typename PreImage, typename Image>
  class Hasher
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

    Hasher(unsigned int width, unsigned int preimage_max_length, unsigned int rate, unsigned int offset)
        : width(width), preimage_max_length(preimage_max_length), rate(rate), offset(offset)
    {
      assert(
        rate * sizeof(PreImage) <= preimage_max_length * sizeof(Image) &&
        "Input rate can not be bigger than preimage max length");
    }

    virtual cudaError_t hash_2d(
      const Matrix<PreImage>* inputs,
      Image* states,
      unsigned int number_of_inputs,
      unsigned int output_len,
      uint64_t number_of_rows,
      const device_context::DeviceContext& ctx) const
    {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Absorb 2d is not implemented for this hash");
      return cudaError_t::cudaSuccess;
    };

    virtual cudaError_t compress_and_inject(
      const Matrix<PreImage>* matrices_to_inject,
      unsigned int number_of_inputs,
      uint64_t number_of_rows,
      const Image* prev_layer,
      Image* next_layer,
      unsigned int digest_elements,
      const device_context::DeviceContext& ctx) const
    {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Compress and inject is not implemented for this hash");
      return cudaError_t::cudaSuccess;
    }

    /// @param input pointer to input allocated on-device
    /// @param out pointer to output allocated on-device
    cudaError_t compress_many(
      const Image* input,
      Image* out,
      unsigned int number_of_states,
      unsigned int output_len,
      const HashConfig& cfg) const
    {
      return hash_many((const PreImage*)input, out, number_of_states, width, output_len, cfg);
    }

    virtual cudaError_t run_hash_many_kernel(
      const PreImage* input,
      Image* output,
      unsigned int number_of_states,
      unsigned int input_len,
      unsigned int output_len,
      const device_context::DeviceContext& ctx) const
    {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Hash many kernel is not implemented for this hash");
      return cudaError_t::cudaSuccess;
    };

    cudaError_t hash_many(
      const PreImage* input,
      Image* output,
      unsigned int number_of_states,
      unsigned int input_len,
      unsigned int output_len,
      const HashConfig& cfg) const
    {
      const PreImage* d_input;
      PreImage* d_alloc_input;
      Image* d_output;
      if (!cfg.are_inputs_on_device) {
        CHK_IF_RETURN(cudaMallocAsync(&d_alloc_input, number_of_states * input_len * sizeof(PreImage), cfg.ctx.stream));
        CHK_IF_RETURN(cudaMemcpyAsync(
          d_alloc_input, input, number_of_states * input_len * sizeof(PreImage), cudaMemcpyHostToDevice,
          cfg.ctx.stream));
        d_input = d_alloc_input;
      } else {
        d_input = input;
      }

      if (!cfg.are_outputs_on_device) {
        CHK_IF_RETURN(cudaMallocAsync(&d_output, number_of_states * output_len * sizeof(Image), cfg.ctx.stream));
      } else {
        d_output = output;
      }

      CHK_IF_RETURN(run_hash_many_kernel(d_input, d_output, number_of_states, input_len, output_len, cfg.ctx));

      if (!cfg.are_inputs_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_input, cfg.ctx.stream)); }
      if (!cfg.are_outputs_on_device) {
        CHK_IF_RETURN(cudaMemcpyAsync(
          output, d_output, number_of_states * output_len * sizeof(Image), cudaMemcpyDeviceToHost, cfg.ctx.stream));
        CHK_IF_RETURN(cudaFreeAsync(d_output, cfg.ctx.stream));
      }

      if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));

      return CHK_LAST();
    };
  };
} // namespace hash

#endif