#pragma once
#ifndef POSEIDON_H
#define POSEIDON_H

#include <cstdint>
#include <stdexcept>
#include "../../utils/device_context.cuh"
#include "../../utils/error_handler.cuh"

namespace poseidon {
#define FIRST_FULL_ROUNDS  true
#define SECOND_FULL_ROUNDS false
  
  enum ARITY {
    TWO,
    FOUR,
    EIGHT,
    ELEVEN   
  };

  const int FULL_ROUNDS_DEFAULT = 4;

  template <typename S>
  struct PoseidonConstants {
    int partial_rounds;
    int full_rounds_half;
    S *round_constants;
    S *mds_matrix;
    S *non_sparse_matrix;
    S *sparse_matrices;
    S domain_tag;
  };

  template <typename S> PoseidonConstants<S> optimizedPoseidonConstants(cudaStream_t stream);

  /// This class describes the logic of calculating CUDA kernels parameters
  /// such as the number of threads and the number of blocks
  struct PoseidonKernelsConfiguration
  {
    int t;
    int number_of_threads;
    int hashes_per_block;
    int singlehash_block_size;

    int number_of_full_blocks(size_t number_of_states)
    {
      int total_number_of_threads = number_of_states * t;
      return total_number_of_threads / number_of_threads +
              static_cast<bool>(total_number_of_threads % number_of_threads);
    }

    int number_of_singlehash_blocks(size_t number_of_states)
    {
      return number_of_states / singlehash_block_size + static_cast<bool>(number_of_states % singlehash_block_size);
    }
  };

  PoseidonKernelsConfiguration default_poseidon_kernels_configuration(int t) {
    // The logic behind this is that 1 thread only works on 1 element
    // We have {t} elements in each state, and {number_of_states} states total

    int number_of_threads = (256 / t) * t;
    int hashes_per_block = number_of_threads / t;

    // The partial rounds operates on the whole state, so we define
    // the parallelism params for processing a single hash preimage per thread
    int singlehash_block_size = 128;

    PoseidonKernelsConfiguration config = {
      t,
      number_of_threads,
      hashes_per_block,
      singlehash_block_size,
    };

    return config;
  }

  struct PoseidonConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    PoseidonKernelsConfiguration kernel_cfg;
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool input_is_a_state;
    bool is_async;              /**< Whether to run the NTT asyncronously. If set to `true`, the NTT function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                 *   function will block the current CPU thread. */
    bool loop_results;
  };

  PoseidonConfig default_poseidon_config(int t) {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    PoseidonKernelsConfiguration kernel_cfg = default_poseidon_kernels_configuration(t);
    PoseidonConfig config = {
      ctx,        // ctx
      kernel_cfg, // kernel_cfg
      false,      // are_inputes_on_device
      false,      // are_outputs_on_device
      false,      // input_is_a_state
      false,      // is_async
      false,      // loop_results
    };
    return config;
  }

  template <typename S>
  PoseidonConstants<S> load_optimized_poseidon_constants(int t, cudaStream_t stream);

  // Compute the poseidon hash over a sequence of preimages
  ///
  ///=====================================================
  /// # Arguments
  /// * `states`  - a device pointer to the states memory. Expected to be of size `number_of_states * t` elements.
  /// States should contain the leaves values
  /// * `number_of_states`  - number of preimages number_of_states. Each block is of size t
  /// * `out` - a device pointer to the digests memory. Expected to be of size `sum(arity ^ (i)) for i in
  /// [0..height-1]`
  /// * `hash_type`  - this will determine the domain_tag value
  /// * `stream` - a cuda stream to run the kernels
  /// * `aligned` - if set to `true`, the algorithm expects the states to contain leaves in an aligned form
  /// * `loop_results` - if set to `true`, the resulting hash will be also copied into the states memory in aligned
  /// form.
  ///
  /// Aligned form (for arity = 2):
  /// [0, X1, X2, 0, X3, X4, ...]
  ///
  /// Not aligned form (for arity = 2) (you will get this format
  ///                                   after copying leaves with cudaMemcpy2D):
  /// [X1, X2, 0, X3, X4, 0]
  /// Note: elements denoted by 0 doesn't need to be set to 0, the algorithm
  /// will replace them with domain tags.
  ///
  /// # Algorithm
  /// The function will split large trees into many subtrees of size that will fit `STREAM_CHUNK_SIZE`.
  /// The subtrees will be constructed in streams pool. Each stream will handle a subtree
  /// After all subtrees are constructed - the function will combine the resulting sub-digests into the final top-tree
  ///======================================================
  template <typename S, int T>
  cudaError_t poseidon_hash(
    S* input,
    S* output,
    size_t number_of_states,
    PoseidonConstants<S>& constants,
    PoseidonConfig& config
  );

  extern "C" cudaError_t PoseidonHash(
    curve_config::scalar_t* input,
    curve_config::scalar_t* output,
    int number_of_states,
    ARITY arity,
    PoseidonConstants<curve_config::scalar_t>& constants,
    PoseidonConfig& config
  ) {
    switch (arity) {
      case TWO:
        return poseidon_hash<curve_config::scalar_t, 3>(input, output, number_of_states, constants, config);
      case FOUR:
        return poseidon_hash<curve_config::scalar_t, 5>(input, output, number_of_states, constants, config);
      case EIGHT:
        return poseidon_hash<curve_config::scalar_t, 9>(input, output, number_of_states, constants, config);
      case ELEVEN:
        return poseidon_hash<curve_config::scalar_t, 12>(input, output, number_of_states, constants, config);
      default:
        throw std::runtime_error("invalid arity");
    }
  }
} // namespace poseidon

#endif