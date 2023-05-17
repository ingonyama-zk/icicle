#pragma once
#include "../../curves/curve_config.cuh"
#include "constants.cuh"

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
#include <iomanip>
#include <string>
#include <sstream>
#include <chrono>

__host__ void print_scalar_t(scalar_t element) {
  uint32_t * limbs = element.export_limbs();

  std::stringstream hex_string;
  hex_string << std::hex << std::setfill('0');

  for (int i = 0; i < scalar_t::TLC; i++) {
      hex_string << std::setw(8) << limbs[i];
  }

  std::cout << "0x" << hex_string.str() << std::endl;
}

__host__ void print_buffer_from_cuda(scalar_t * device_ptr, size_t size) {
  scalar_t * buffer = static_cast< scalar_t * >(malloc(size * sizeof(scalar_t)));
  cudaMemcpy(buffer, device_ptr, size * sizeof(scalar_t), cudaMemcpyDeviceToHost);

  std::cout << "Start print" << std::endl;
  for(int i = 0; i < size / 9; i++) {
    std::cout << "State #" << i << std::endl;
    for (int j = 0; j < 9; j++) {
        print_scalar_t(buffer[i * 9 + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  free(buffer);
}
#endif

#ifdef DEBUG
__device__ void print_scalar(scalar_t element, int data) {
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

struct PoseidonConfiguration {
    uint partial_rounds, full_rounds_half, t;
    scalar_t * round_constants, * mds_matrix, * non_sparse_matrix, *sparse_matrices;
};

class Poseidon {
  public:
    uint t;
    PoseidonConfiguration config;

    enum HashType {
        ConstInputLen,
        MerkleTree,
    };

    Poseidon(const uint arity) {
        t = arity + 1;
        this->config.t = t;

        // Pre-calculate domain tags
        // Domain tags will vary for different applications of Poseidon
        uint32_t tree_domain_tag_value = 1;
        tree_domain_tag_value = (tree_domain_tag_value << arity) - tree_domain_tag_value;
        tree_domain_tag = scalar_t::from(tree_domain_tag_value);

        const_input_no_pad_domain_tag = scalar_t::one();

        // TO-DO: implement binary shifts for scalar type
        // const_input_no_pad_domain_tag = scalar_t::one() << 64;
        // const_input_no_pad_domain_tag *= scalar_t::from(arity);

        get_round_numbers(arity, &this->config.partial_rounds, &this->config.full_rounds_half);

        uint round_constants_len = t * this->config.full_rounds_half * 2 + this->config.partial_rounds;
        uint mds_matrix_len = t * t;
        uint sparse_matrices_len = (t * 2 - 1) * this->config.partial_rounds;

        // All the constants are stored in a single file
        scalar_t * constants = load_constants(arity);

        scalar_t * mds_offset = constants + round_constants_len;
        scalar_t * non_sparse_offset = mds_offset + mds_matrix_len;
        scalar_t * sparse_matrices_offset = non_sparse_offset + mds_matrix_len;

        #if !defined(__CUDA_ARCH__) && defined(DEBUG)
        std::cout << "P: " << this->config.partial_rounds << " F: " << this->config.full_rounds_half << std::endl;
        #endif

        // Allocate the memory for constants
        cudaMalloc(&this->config.round_constants, sizeof(scalar_t) * round_constants_len);
        cudaMalloc(&this->config.mds_matrix, sizeof(scalar_t) * mds_matrix_len);
        cudaMalloc(&this->config.non_sparse_matrix, sizeof(scalar_t) * mds_matrix_len);
        cudaMalloc(&this->config.sparse_matrices, sizeof(scalar_t) * sparse_matrices_len);

        // Copy the constants to device
        cudaMemcpy(this->config.round_constants, constants,
                sizeof(scalar_t) * round_constants_len,
                cudaMemcpyHostToDevice);
        cudaMemcpy(this->config.mds_matrix, mds_offset,
                sizeof(scalar_t) * mds_matrix_len,
                cudaMemcpyHostToDevice);
        cudaMemcpy(this->config.non_sparse_matrix, non_sparse_offset,
                sizeof(scalar_t) * mds_matrix_len,
                cudaMemcpyHostToDevice);
        cudaMemcpy(this->config.sparse_matrices, sparse_matrices_offset,
                sizeof(scalar_t) * sparse_matrices_len,
                cudaMemcpyHostToDevice);

        free(constants);
    }

    ~Poseidon() {
        cudaFree(this->config.round_constants);
        cudaFree(this->config.mds_matrix);
        cudaFree(this->config.non_sparse_matrix);
        cudaFree(this->config.sparse_matrices);
    }

    // Hash multiple preimages in parallel
    void hash_blocks(const scalar_t * inp, size_t blocks, scalar_t * out, HashType hash_type);

  private:
    scalar_t tree_domain_tag, const_input_no_pad_domain_tag;
};
