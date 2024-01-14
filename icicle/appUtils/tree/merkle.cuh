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

    struct MerkleConfig {
        device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
        bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
        bool is_async;              /**< Whether to run the NTT asyncronously. If set to `true`, the NTT function will be
                                    *   non-blocking and you'd need to synchronize it explicitly by running
                                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                    *   function will block the current CPU thread. */
    };

    MerkleConfig default_merkle_config() {
        device_context::DeviceContext ctx = device_context::get_default_device_context();
        MerkleConfig config = {
            ctx,        // ctx
            false,      // are_inputes_on_device
            false,      // is_async
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
    template <typename S>
    cudaError_t build_merkle_tree(const S* leaves, S* digests, uint32_t height, PoseidonConstants<S>& poseidon, MerkleConfig& config);
}

#endif