#pragma once
#ifndef MERKLE_H
#define MERKLE_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "utils/utils.h"
#include "hash/hash.cuh"
#include "matrix/matrix.cuh"

#include <vector>
#include <numeric>
#include <iostream>
#include <math.h>

using namespace hash;
using matrix::Matrix;

/**
 * @namespace merkle_tree
 * Implementation of the [Merkle tree](https://en.wikipedia.org/wiki/Merkle_tree) builder,
 * parallelized for the use on GPU
 */
namespace merkle_tree {
  static constexpr size_t GIGA = 1024 * 1024 * 1024;

  /// Bytes per stream
  static constexpr uint64_t STREAM_CHUNK_SIZE = GIGA;

  /// Flattens the tree digests and sum them up to get
  /// the memory needed to contain all the digests
  static size_t get_digests_len(uint32_t height, uint32_t arity, uint32_t digest_elements)
  {
    size_t digests_len = 0;
    size_t row_length = digest_elements;
    for (int i = 0; i <= height; i++) {
      digests_len += row_length;
      row_length *= arity;
    }

    return digests_len;
  }

  template <typename T>
  void swap(T** r, T** s)
  {
    T* t = *r;
    *r = *s;
    *s = t;
  }

  static unsigned int get_height(uint64_t number_of_elements)
  {
    unsigned int height = 0;
    while (number_of_elements >>= 1)
      ++height;
    return height;
  }

  /**
   * @struct TreeBuilderConfig
   * Struct that encodes various Tree builder parameters.
   */
  struct TreeBuilderConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    unsigned int arity;
    unsigned int
      keep_rows; /**< How many rows of the Merkle tree rows should be written to output. '0' means all of them */
    unsigned int
      digest_elements;         /** @param digest_elements the size of output for each bottom layer hash and compression.
                                *  Will also be equal to the size of the root of the tree. Default value 1 */
    bool sort_inputs;
    bool are_inputs_on_device; /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool
      are_outputs_on_device; /**< True if outputs are on device and false if they're on host. Default value: false. */
    bool is_async; /**< Whether to run the tree builder asynchronously. If set to `true`, the build_merkle_tree
                    *   function will be non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */
  };

  static TreeBuilderConfig
  default_merkle_config(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    TreeBuilderConfig config = {
      ctx,   // ctx
      2,     // arity
      0,     // keep_rows
      1,     // digest_elements
      true,  // sort_inputs
      false, // are_inputes_on_device
      false, // are_outputs_on_device
      false, // is_async
    };
    return config;
  }

  /**
   * Builds the Merkle tree
   *
   * @param leaves a pointer to the leaves layer. May be allocated on device or on host, regulated by the config
   * Expected to have arity ^ (height) * input_block_len elements
   * @param digests a pointer to the digests storage. May only be allocated on the host
   * Expected to have `sum(digests_len * (arity ^ (i))) for i in [0..keep_rows]`
   * @param height the height of the merkle tree
   * @param input_block_len the size of input vectors at the bottom layer of the tree
   * # Algorithm
   * The function will split large tree into many subtrees of size that will fit `STREAM_CHUNK_SIZE`.
   * Each subtree is build in it's own stream (there is a maximum number of streams)
   * After all subtrees are constructed - the function will combine the resulting sub-digests into the final top-tree
   */
  template <typename Leaf, typename Digest>
  cudaError_t build_merkle_tree(
    const Leaf* inputs,
    Digest* digests,
    unsigned int height,
    unsigned int input_block_len,
    const Hasher<Leaf, Digest>& compression,
    const Hasher<Leaf, Digest>& bottom_layer,
    const TreeBuilderConfig& config);

  template <typename Leaf, typename Digest>
  cudaError_t mmcs_commit(
    const Matrix<Leaf>* inputs,
    const unsigned int number_of_inputs,
    Digest* digests,
    const Hasher<Leaf, Digest>& hasher,
    const Hasher<Leaf, Digest>& compression,
    const TreeBuilderConfig& tree_config);
} // namespace merkle_tree

#endif