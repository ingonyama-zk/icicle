#pragma once
#ifndef POSEIDON2_H
#define POSEIDON2_H

#include <cstdint>
#include <stdexcept>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"

/**
 * @namespace poseidon2
 * Implementation of the [Poseidon2 hash function](https://eprint.iacr.org/2019/458.pdf)
 * Specifically, the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/)
 */
namespace poseidon2 {
  /**
   * For most of the Poseidon2 configurations this is the case
   */
  const int EXTERNAL_ROUNDS_DEFAULT = 8;

  enum MdsType {
    DEFAULT,
    PLONKY
  };

  enum PoseidonMode {
    COMPRESSION,
    PERMUTATION,
  };

  /**
   * @struct Poseidon2Constants
   * This constants are enough to define a Poseidon2 instantce
   * @param round_constants A pointer to round constants allocated on the device
   * @param mds_matrix A pointer to an mds matrix allocated on the device
   * @param non_sparse_matrix A pointer to non sparse matrix allocated on the device
   * @param sparse_matrices A pointer to sparse matrices allocated on the device
   */
  template <typename S>
  struct Poseidon2Constants {
    int width;
    int alpha;
    int internal_rounds;
    int external_rounds;
    S* round_constants = nullptr;
    S* internal_matrix_diag = nullptr;
  };

  /**
   * @struct Poseidon2Config
   * Struct that encodes various Poseidon2 parameters.
   */
  struct Poseidon2Config {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    bool are_states_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    PoseidonMode mode;
    MdsType mds_type;
    int output_index;
    bool loop_state;            /**< If true, hash results will also be copied in the input pointer in aligned format */
    bool is_async; /**< Whether to run the Poseidon2 asynchronously. If set to `true`, the poseidon_hash function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the poseidon_hash
                    *   function will block the current CPU thread. */
  };

  static Poseidon2Config default_poseidon2_config(
    int t, const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    Poseidon2Config config = {
      ctx,   // ctx
      false, // are_states_on_device
      false, // are_outputs_on_device
      PoseidonMode::COMPRESSION,
      MdsType::DEFAULT,
      1, // output_index
      false, // loop_state
      false, // is_async
    };
    return config;
  }

  template <typename S>
  cudaError_t create_optimized_poseidon2_constants(
    int width,
    int alpha,
    int internal_rounds,
    int external_rounds,
    const S* round_constants,
    const S* internal_matrix_diag,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<S>* poseidon_constants);

  /**
   * Loads pre-calculated optimized constants, moves them to the device
   */
  template <typename S>
  cudaError_t
  init_optimized_poseidon2_constants(int width, device_context::DeviceContext& ctx, Poseidon2Constants<S>* constants);

  /**
   * Compute the poseidon hash over a sequence of preimages.
   * Takes {number_of_states * (T-1)} elements of input and computes {number_of_states} hash images
   * @param T size of the poseidon state, should be equal to {arity + 1}
   * @param states a pointer to the input data. May be allocated on device or on host, regulated
   * by the config. May point to a string of preimages or a string of states filled with preimages.
   * @param output a pointer to the output data. May be allocated on device or on host, regulated
   * by the config. Must be at least of size [number_of_states](@ref number_of_states)
   * @param number_of_states number of input blocks of size T-1 (arity)
   */
  template <typename S, int T>
  cudaError_t poseidon2_hash(
    S* states, S* output, size_t number_of_states, const Poseidon2Constants<S>& constants, const Poseidon2Config& config);
} // namespace poseidon2

#endif