#pragma once
#ifndef HASH_H
#define HASH_H

template <typename T, int WIDTH>
class Permutation {
    virtual void permute_many(
        const T* states,
        T* output,
        unsigned int number_of_states
    );
};

template <typename T, int WIDTH>
class CompressionHasher {
    virtual void compress_many(
        const T* input,
        T* output,
        unsigned int number_of_states
    );
};

template <typename PI, typename I, int WIDTH, int RATE>
class SpongeHasher {
    virtual void hash_many(
        const PI* input,
        I* output,
        unsigned int number_of_states
    );
};

#endif