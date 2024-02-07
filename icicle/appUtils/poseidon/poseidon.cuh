#pragma once
#ifndef POSEIDON_H
#define POSEIDON_H

#include <cstdint>
#include <stdexcept>
#include "utils/device_context.cuh"
#include "curves/curve_config.cuh"
#include "utils/error_handler.cuh"
#include "utils/utils.h"

/**
 * @namespace poseidon
 * Implementation of the [Poseidon hash function](https://eprint.iacr.org/2019/458.pdf)
 * Specifically, the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/)
 */
namespace poseidon {
#define FIRST_FULL_ROUNDS  true
#define SECOND_FULL_ROUNDS false

  /**
   * For most of the Poseidon configurations this is the case
   * To-do: Add support for different full rounds numbers
   */
  const int FULL_ROUNDS_DEFAULT = 4;

  /**
   * @struct PoseidonConstants
   * This constants are enough to define a Poseidon instantce
   * @param round_constants A pointer to round constants allocated on the device
   * @param mds_matrix A pointer to an mds matrix allocated on the device
   * @param non_sparse_matrix A pointer to non sparse matrix allocated on the device
   * @param sparse_matrices A pointer to sparse matrices allocated on the device
   */
  template <typename S>
  struct PoseidonConstants {
    int arity;
    int partial_rounds;
    int full_rounds_half;
    S* round_constants = nullptr;
    S* mds_matrix = nullptr;
    S* non_sparse_matrix = nullptr;
    S* sparse_matrices = nullptr;
    S domain_tag;
  };

  /**
   * @class PoseidonKernelsConfiguration
   * Describes the logic of deriving CUDA kernels parameters
   * such as the number of threads and the number of blocks
   */
  template <int T>
  class PoseidonKernelsConfiguration
  {
  public:
    // The logic behind this is that 1 thread only works on 1 element
    // We have {T} elements in each state, and {number_of_states} states total
    static const int number_of_threads = 256 / T * T;

    // The partial rounds operates on the whole state, so we define
    // the parallelism params for processing a single hash preimage per thread
    static const int singlehash_block_size = 128;

    static const int hashes_per_block = number_of_threads / T;

    static int number_of_full_blocks(size_t number_of_states)
    {
      int total_number_of_threads = number_of_states * T;
      return total_number_of_threads / number_of_threads +
             static_cast<bool>(total_number_of_threads % number_of_threads);
    }

    static int number_of_singlehash_blocks(size_t number_of_states)
    {
      return number_of_states / singlehash_block_size + static_cast<bool>(number_of_states % singlehash_block_size);
    }
  };

  template <int T>
  using PKC = PoseidonKernelsConfiguration<T>;

  /**
   * @struct NTTConfig
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

  template <typename S>
  PoseidonConfig default_poseidon_config(int t)
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
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

  /**
   * Loads pre-calculated optimized constants, moves them to the device
   */
  template <typename S>
  cudaError_t init_optimized_poseidon_constants(device_context::DeviceContext& ctx, PoseidonConstants<S>* constants);

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
} // namespace poseidon

#endif