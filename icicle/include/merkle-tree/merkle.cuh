#pragma once
#ifndef MERKLE_H
#define MERKLE_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"
#include "hash/hash.cuh"

#include <iostream>
#include <math.h>

/**
 * @namespace merkle_tree
 * Implementation of the [Merkle tree](https://en.wikipedia.org/wiki/Merkle_tree) builder,
 * parallelized for the use on GPU
 */
namespace merkle_tree {
  static constexpr size_t GIGA = 1024 * 1024 * 1024;

  /// Bytes per stream
  static constexpr size_t STREAM_CHUNK_SIZE = 1024 * 1024 * 1024;

  /**
   * @struct TreeBuilderConfig
   * Struct that encodes various Tree builder parameters.
   */
  struct TreeBuilderConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    int arity;
    int keep_rows; /**< How many rows of the Merkle tree rows should be written to output. '0' means all of them */
    bool are_inputs_on_device; /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool is_async; /**< Whether to run the tree builder asynchronously. If set to `true`, the build_merkle_tree
                    *   function will be non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */
  };

  static TreeBuilderConfig
  default_merkle_config(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    TreeBuilderConfig config = {
      ctx, // ctx
      2,
      0,     // keep_rows
      false, // are_inputes_on_device
      false, // is_async
    };
    return config;
  }

  /**
   * Builds the Merkle tree
   *
   * @param leaves a pointer to the leaves layer. May be allocated on device or on host, regulated by the config
   * Expected to have arity ^ (height - 1) elements
   * @param digests a pointer to the digests storage. May only be allocated on the host
   * Expected to have `sum(arity ^ (i)) for i in [0..height-1]`
   * @param height the height of the merkle tree
   * # Algorithm
   * The function will split large tree into many subtrees of size that will fit `STREAM_CHUNK_SIZE`.
   * Each subtree is build in it's own stream (there is a maximum number of streams)
   * After all subtrees are constructed - the function will combine the resulting sub-digests into the final top-tree
   */
  template <typename Leaf, typename Digest>
  cudaError_t build_merkle_tree(
    const Leaf* leaves,
    Digest* digests,
    uint32_t height,
    uint32_t arity,
    const SpongeHasher<Leaf, Digest>& sponge,
    const CompressionHasher<Digest>& compression,
    const SpongeConfig& sponge_config,
    const TreeBuilderConfig& config);
} // namespace merkle_tree

#endif