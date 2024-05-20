#pragma once
#ifndef HASH_H
#define HASH_H

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

template <typename Image>
class Permutation {
  public:
    virtual cudaError_t permute_many(
        const Image* states,
        Image* output,
        unsigned int number_of_states,
        device_context::DeviceContext& ctx
    ) = 0;
};

template <typename Image>
class CompressionHasher {
  public:
    virtual cudaError_t compress_many(
        const Image* states,
        Image* output,
        unsigned int number_of_states,
        unsigned int rate,
        device_context::DeviceContext& ctx,
        unsigned int offset=0,
        Image* perm_output=nullptr
    ) = 0;
};

// template <typename PreImage, typename Image, int WIDTH, int RATE>
// class SpongeHasher {
//     constexpr static int capacity = WIDTH - RATE;

//     virtual cudaError_t hash_many(
//         const PreImage** input,
//         Image* output,
//         unsigned int number_of_states,
//         device_context::DeviceContext& ctx
//     );
// };

#endif