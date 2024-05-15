#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

template <typename Image, int WIDTH>
class Permutation {
    virtual cudaError_t permute_many(
        const Image* states,
        Image* output,
        unsigned int number_of_states
        DeviceContext& ctx,
        bool is_async
    );
};

template <typename Image, int WIDTH, int RATE>
class CompressionHasher {
    virtual cudaError_t compress_many(
        const Image* states,
        Image* output,
        unsigned int number_of_states,
        DeviceContext& ctx,
        bool is_async,
        Image* perm_output=nullptr
    );
};

template <typename PreImage, typename Image, int WIDTH, int RATE>
class SpongeHasher {
    constexpr static int capacity = WIDTH - RATE;

    virtual cudaError_t hash_many(
        const PreImage* input,
        Image* output,
        unsigned int number_of_states
        DeviceContext& ctx,
        bool is_async
    );
};

#endif