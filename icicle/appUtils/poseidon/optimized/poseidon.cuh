#pragma once
#ifndef POSEIDON_OPTIMIZED_H
#define POSEIDON_OPTIMIZED_H

#include "../poseidon.cuh"
#include "../../../curves/curve_config.cuh"

namespace poseidon {
    template <typename S>
    S* load_optimized_constants(const uint32_t arity)
    {
    unsigned char* constants;
    switch (arity) {
    case 2:
        constants = poseidon_constants_2;
        break;
    case 4:
        constants = poseidon_constants_4;
        break;
    case 8:
        constants = poseidon_constants_8;
        break;
    case 11:
        constants = poseidon_constants_11;
        break;
    default:
        throw std::invalid_argument("unsupported arity");
    }
    return reinterpret_cast<S*>(constants);
    }

    template <typename S> __global__ void prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, const PoseidonConfiguration<S> config, bool aligned);
    template <typename S> __global__ void get_hash_results(S* states, size_t number_of_states, S* out, int t);
    template <typename S> __global__ void copy_recursive(S * state, size_t number_of_states, S * out, int t);
    template <typename S> __global__ void full_rounds(S* states, size_t number_of_states, size_t rc_offset, bool first_half, const PoseidonConfiguration<S> config);
    template <typename S> __global__ void partial_rounds(S* states, size_t number_of_states, size_t rc_offset, const PoseidonConfiguration<S> config);

    template <typename S>
    class OptimizedPoseidon: public Poseidon<S>
    {
    public:
    PoseidonConfiguration<S> config;
    ParallelPoseidonConfiguration kernel_params;

    OptimizedPoseidon(const uint32_t arity, cudaStream_t stream) : Poseidon<S>(arity), kernel_params(arity + 1) {
        config.t = arity + 1;
        this->stream = stream;

        config.full_rounds_half = FULL_ROUNDS_DEFAULT;
        config.partial_rounds = partial_rounds_number_from_arity(arity);

        uint32_t round_constants_len = config.t * config.full_rounds_half * 2 + config.partial_rounds;
        uint32_t mds_matrix_len = config.t * config.t;
        uint32_t sparse_matrices_len = (config.t * 2 - 1) * config.partial_rounds;

        // All the constants are stored in a single file
        S* constants = load_optimized_constants<S>(arity);

        S* mds_offset = constants + round_constants_len;
        S* non_sparse_offset = mds_offset + mds_matrix_len;
        S* sparse_matrices_offset = non_sparse_offset + mds_matrix_len;

    #if !defined(__CUDA_ARCH__) && defined(DEBUG)
        for (int i = 0; i < mds_matrix_len; i++) {
        std::cout << mds_offset[i] << std::endl;
        }
        std::cout << "P: " << config.partial_rounds << " F: " << config.full_rounds_half << std::endl;
    #endif

        // Create streams for copying constants
        cudaStream_t stream_copy_round_constants, stream_copy_mds_matrix, stream_copy_non_sparse,
        stream_copy_sparse_matrices;
        cudaStreamCreate(&stream_copy_round_constants);
        cudaStreamCreate(&stream_copy_mds_matrix);
        cudaStreamCreate(&stream_copy_non_sparse);
        cudaStreamCreate(&stream_copy_sparse_matrices);

        // Create events for copying constants
        cudaEvent_t event_copied_round_constants, event_copy_mds_matrix, event_copy_non_sparse, event_copy_sparse_matrices;
        cudaEventCreateWithFlags(&event_copied_round_constants, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_copy_mds_matrix, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_copy_non_sparse, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_copy_sparse_matrices, cudaEventDisableTiming);

        // Malloc memory for copying constants
        cudaMallocAsync(&config.round_constants, sizeof(S) * round_constants_len, stream_copy_round_constants);
        cudaMallocAsync(&config.mds_matrix, sizeof(S) * mds_matrix_len, stream_copy_mds_matrix);
        cudaMallocAsync(&config.non_sparse_matrix, sizeof(S) * mds_matrix_len, stream_copy_non_sparse);
        cudaMallocAsync(&config.sparse_matrices, sizeof(S) * sparse_matrices_len, stream_copy_sparse_matrices);

        // Copy constants
        cudaMemcpyAsync(
        config.round_constants, constants, sizeof(S) * round_constants_len, cudaMemcpyHostToDevice,
        stream_copy_round_constants);
        cudaMemcpyAsync(
        config.mds_matrix, mds_offset, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice, stream_copy_mds_matrix);
        cudaMemcpyAsync(
        config.non_sparse_matrix, non_sparse_offset, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice,
        stream_copy_non_sparse);
        cudaMemcpyAsync(
        config.sparse_matrices, sparse_matrices_offset, sizeof(S) * sparse_matrices_len, cudaMemcpyHostToDevice,
        stream_copy_sparse_matrices);

        // Record finished copying event for streams
        cudaEventRecord(event_copied_round_constants, stream_copy_round_constants);
        cudaEventRecord(event_copy_mds_matrix, stream_copy_mds_matrix);
        cudaEventRecord(event_copy_non_sparse, stream_copy_non_sparse);
        cudaEventRecord(event_copy_sparse_matrices, stream_copy_sparse_matrices);

        // Main stream waits for copying to finish
        cudaStreamWaitEvent(stream, event_copied_round_constants);
        cudaStreamWaitEvent(stream, event_copy_mds_matrix);
        cudaStreamWaitEvent(stream, event_copy_non_sparse);
        cudaStreamWaitEvent(stream, event_copy_sparse_matrices);
    }

    ~OptimizedPoseidon()
    {
        cudaFreeAsync(config.round_constants, stream);
        cudaFreeAsync(config.mds_matrix, stream);
        cudaFreeAsync(config.non_sparse_matrix, stream);
        cudaFreeAsync(config.sparse_matrices, stream);
    }

    void prepare_states(S * states, size_t number_of_states, S domain_tag, bool aligned) override {
        prepare_poseidon_states<<<
        kernel_params.number_of_full_blocks(number_of_states),
        kernel_params.number_of_threads,
        0,
        stream
        >>>(states, number_of_states, domain_tag, config, aligned);
    }

    void process_results(S * states, size_t number_of_states, S * out, bool loop_results) override {
        get_hash_results<<<
        kernel_params.number_of_singlehash_blocks(number_of_states),
        kernel_params.singlehash_block_size,
        0,
        stream
        >>> (states, number_of_states, out, config.t);

        if (loop_results) {
        copy_recursive <<<
        kernel_params.number_of_singlehash_blocks(number_of_states),
        kernel_params.singlehash_block_size,
            0,
            stream
        >>> (states, number_of_states, out, config.t);
        }
    }

    void permute_many(S * states, size_t number_of_states, cudaStream_t stream) override {
        size_t rc_offset = 0;
        
        // execute half full rounds
        full_rounds<<<
        kernel_params.number_of_full_blocks(number_of_states),
        kernel_params.number_of_threads,
        sizeof(S) * kernel_params.hashes_per_block * config.t,
        stream
        >>>(states, number_of_states, rc_offset, FIRST_FULL_ROUNDS, config);
        rc_offset += config.t * (config.full_rounds_half + 1);

    #if !defined(__CUDA_ARCH__) && defined(DEBUG)
        cudaStreamSynchronize(stream);
        std::cout << "Full rounds 1. RCOFFSET: " << rc_offset << std::endl;
        print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
    #endif

        // execute partial rounds
        partial_rounds<<<
        kernel_params.number_of_singlehash_blocks(number_of_states),
        kernel_params.singlehash_block_size,
        0,
        stream
        >>>(states, number_of_states, rc_offset, config);
        rc_offset += config.partial_rounds;

    #if !defined(__CUDA_ARCH__) && defined(DEBUG)
        cudaStreamSynchronize(stream);
        std::cout << "Partial rounds. RCOFFSET: " << rc_offset << std::endl;
        print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
    #endif

        // execute half full rounds
        full_rounds<<<
        kernel_params.number_of_full_blocks(number_of_states),
        kernel_params.number_of_threads,
        sizeof(S) * kernel_params.hashes_per_block * config.t,
        stream
        >>>(states, number_of_states, rc_offset, SECOND_FULL_ROUNDS, config);

    #if !defined(__CUDA_ARCH__) && defined(DEBUG)
        cudaStreamSynchronize(stream);
        std::cout << "Full rounds 2. RCOFFSET: " << rc_offset << std::endl;
        print_buffer_from_cuda<S>(states, number_of_states * config.t, config.t);
    #endif
    }
    
    private:
    cudaStream_t stream;
    };
}

#endif