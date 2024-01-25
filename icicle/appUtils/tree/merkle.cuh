#pragma once
#ifndef MERKLE_H
#define MERKLE_H

#include "../../utils/device_context.cuh"
#include "../../utils/error_handler.cuh"
#include "../poseidon/poseidon.cuh"

#include <iostream>
#include <math.h>

using namespace poseidon;

namespace merkle {
  static constexpr size_t GIGA = 1024 * 1024 * 1024;

  /// Bytes per stream
  static constexpr size_t STREAM_CHUNK_SIZE = 1024 * 1024 * 1024;

  struct MerkleConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    int keep_rows;
    bool are_inputs_on_device; /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool is_async;             /**< Whether to run the NTT asyncronously. If set to `true`, the NTT function will be
                                *   non-blocking and you'd need to synchronize it explicitly by running
                                *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                *   function will block the current CPU thread. */
  };

  MerkleConfig default_merkle_config()
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    MerkleConfig config = {
      ctx,   // ctx
      0,     // keep_rows
      false, // are_inputes_on_device
      false, // is_async
    };
    return config;
  }

  /// Constructs the merkle tree
  ///
  ///=====================================================
  /// # Arguments
  /// * `leaves`  - a host pointer to the leaves array. Expected to have arity ^ (height - 1) elements
  /// * `digests` - a host pointer to write digests to. Expected to have `sum(arity ^ (i)) for i in [0..height-1]`
  /// elements
  /// * `height`  - the height of a tree
  /// * `poseidon` - an instance of the poseidon hasher
  /// * `stream` - a cuda stream for top-level operations
  ///
  /// # Algorithm
  /// The function will split large trees into many subtrees of size that will fit `STREAM_CHUNK_SIZE`.
  /// The subtrees will be constructed in streams pool. Each stream will handle a subtree
  /// After all subtrees are constructed - the function will combine the resulting sub-digests into the final top-tree
  ///======================================================
  template <typename S, int T>
  cudaError_t
  build_merkle_tree(const S* leaves, S* digests, uint32_t height, PoseidonConstants<S>& poseidon, MerkleConfig& config);

  extern "C" cudaError_t BuildMerkleTree(
    const curve_config::scalar_t* leaves,
    curve_config::scalar_t* digests,
    uint32_t height,
    ARITY arity,
    PoseidonConstants<curve_config::scalar_t>& poseidon,
    MerkleConfig& config)
  {
    switch (arity) {
    case TWO:
      return build_merkle_tree<curve_config::scalar_t, 3>(leaves, digests, height, poseidon, config);
    case FOUR:
      return build_merkle_tree<curve_config::scalar_t, 5>(leaves, digests, height, poseidon, config);
    case EIGHT:
      return build_merkle_tree<curve_config::scalar_t, 9>(leaves, digests, height, poseidon, config);
    case ELEVEN:
      return build_merkle_tree<curve_config::scalar_t, 12>(leaves, digests, height, poseidon, config);
    default:
      throw std::runtime_error("invalid arity");
    }
  }
} // namespace merkle

#endif