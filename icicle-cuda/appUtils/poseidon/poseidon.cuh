#pragma once
#include "constants.cuh"

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <chrono>

#define ARITY 3

template <typename S>
__host__ void print_buffer_from_cuda(S * device_ptr, size_t size) {
  S * buffer = static_cast< S * >(malloc(size * sizeof(S)));
  cudaMemcpy(buffer, device_ptr, size * sizeof(S), cudaMemcpyDeviceToHost);

  std::cout << "Start print" << std::endl;
  for(int i = 0; i < size / ARITY; i++) {
    std::cout << "State #" << i << std::endl;
    for (int j = 0; j < ARITY; j++) {
      std::cout << buffer[i * ARITY + j] << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  free(buffer);
}
#endif

#ifdef DEBUG
template <typename S>
__device__ void print_scalar(S element, int data) {
    printf("D# %d, T# %d: 0x%08x%08x%08x%08x%08x%08x%08x%08x\n",
           data,
           threadIdx.x,
           element.limbs_storage.limbs[0],
           element.limbs_storage.limbs[1],
           element.limbs_storage.limbs[2],
           element.limbs_storage.limbs[3],
           element.limbs_storage.limbs[4],
           element.limbs_storage.limbs[5],
           element.limbs_storage.limbs[6],
           element.limbs_storage.limbs[7]
    );
}
#endif

template <typename S>
struct PoseidonConfiguration {
    uint32_t partial_rounds, full_rounds_half, t;
    S * round_constants, * mds_matrix, * non_sparse_matrix, *sparse_matrices;
};

template <typename S>
class Poseidon {
  public:
    uint32_t t;
    PoseidonConfiguration<S> config;

    enum HashType {
        ConstInputLen,
        MerkleTree,
    };

    Poseidon(const uint32_t arity) {
        t = arity + 1;
        this->config.t = t;

        // Pre-calculate domain tags
        // Domain tags will vary for different applications of Poseidon
        uint32_t tree_domain_tag_value = 1;
        tree_domain_tag_value = (tree_domain_tag_value << arity) - tree_domain_tag_value;
        tree_domain_tag = S::from(tree_domain_tag_value);

        const_input_no_pad_domain_tag = S::one();

        // TO-DO: implement binary shifts for scalar type
        // const_input_no_pad_domain_tag = S::one() << 64;
        // const_input_no_pad_domain_tag *= S::from(arity);

        this->config.full_rounds_half = FULL_ROUNDS_DEFAULT;
        this->config.partial_rounds = partial_rounds_number_from_arity(arity);

        uint32_t round_constants_len = t * this->config.full_rounds_half * 2 + this->config.partial_rounds;
        uint32_t mds_matrix_len = t * t;
        uint32_t sparse_matrices_len = (t * 2 - 1) * this->config.partial_rounds;

        // All the constants are stored in a single file
        S * constants = load_constants<S>(arity);

        S * mds_offset = constants + round_constants_len;
        S * non_sparse_offset = mds_offset + mds_matrix_len;
        S * sparse_matrices_offset = non_sparse_offset + mds_matrix_len;

        #if !defined(__CUDA_ARCH__) && defined(DEBUG)
        std::cout << "P: " << this->config.partial_rounds << " F: " << this->config.full_rounds_half << std::endl;
        #endif

        // Allocate the memory for constants
        cudaMalloc(&this->config.round_constants, sizeof(S) * round_constants_len);
        cudaMalloc(&this->config.mds_matrix, sizeof(S) * mds_matrix_len);
        cudaMalloc(&this->config.non_sparse_matrix, sizeof(S) * mds_matrix_len);
        cudaMalloc(&this->config.sparse_matrices, sizeof(S) * sparse_matrices_len);

        // Copy the constants to device
        cudaMemcpy(this->config.round_constants, constants,
                sizeof(S) * round_constants_len,
                cudaMemcpyHostToDevice);
        cudaMemcpy(this->config.mds_matrix, mds_offset,
                sizeof(S) * mds_matrix_len,
                cudaMemcpyHostToDevice);
        cudaMemcpy(this->config.non_sparse_matrix, non_sparse_offset,
                sizeof(S) * mds_matrix_len,
                cudaMemcpyHostToDevice);
        cudaMemcpy(this->config.sparse_matrices, sparse_matrices_offset,
                sizeof(S) * sparse_matrices_len,
                cudaMemcpyHostToDevice);
    }

    ~Poseidon() {
        cudaFree(this->config.round_constants);
        cudaFree(this->config.mds_matrix);
        cudaFree(this->config.non_sparse_matrix);
        cudaFree(this->config.sparse_matrices);
    }

    // Hash multiple preimages in parallel
    void hash_blocks(const S * inp, size_t blocks, S * out, HashType hash_type);

  private:
    S tree_domain_tag, const_input_no_pad_domain_tag;
};