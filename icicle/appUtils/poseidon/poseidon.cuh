#pragma once
#include "constants.cuh"

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

template <typename S>
__host__ void print_buffer_from_cuda(S* device_ptr, size_t size, size_t t)
{
  S* buffer = static_cast<S*>(malloc(size * sizeof(S)));
  CHK_LOG(checudaMemcpy(buffer, device_ptr, size * sizeof(S), cudaMemcpyDeviceToHost));

  std::cout << "Start print" << std::endl;
  for (int i = 0; i < size / t; i++) {
    std::cout << "State #" << i << std::endl;
    for (int j = 0; j < t; j++) {
      std::cout << buffer[i * t + j] << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  free(buffer);
}
#endif

#ifdef DEBUG
template <typename S>
__device__ void print_scalar(S element, int data)
{
  printf(
    "D# %d, T# %d: 0x%08x%08x%08x%08x%08x%08x%08x%08x\n", data, threadIdx.x, element.limbs_storage.limbs[0],
    element.limbs_storage.limbs[1], element.limbs_storage.limbs[2], element.limbs_storage.limbs[3],
    element.limbs_storage.limbs[4], element.limbs_storage.limbs[5], element.limbs_storage.limbs[6],
    element.limbs_storage.limbs[7]);
}
#endif

template <typename S>
struct PoseidonConfiguration {
  uint32_t partial_rounds, full_rounds_half, t;
  S *round_constants, *mds_matrix, *non_sparse_matrix, *sparse_matrices;
};

template <typename S>
class Poseidon
{
public:
  uint32_t t;
  PoseidonConfiguration<S> config;

  enum HashType {
    ConstInputLen,
    MerkleTree,
  };

  Poseidon(const uint32_t arity, cudaStream_t stream)
  {
    t = arity + 1;
    this->config.t = t;
    this->stream = stream;

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
    S* constants = load_constants<S>(arity);

    S* mds_offset = constants + round_constants_len;
    S* non_sparse_offset = mds_offset + mds_matrix_len;
    S* sparse_matrices_offset = non_sparse_offset + mds_matrix_len;

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
    std::cout << "P: " << this->config.partial_rounds << " F: " << this->config.full_rounds_half << std::endl;
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
    cudaMallocAsync(&this->config.round_constants, sizeof(S) * round_constants_len, stream_copy_round_constants);
    cudaMallocAsync(&this->config.mds_matrix, sizeof(S) * mds_matrix_len, stream_copy_mds_matrix);
    cudaMallocAsync(&this->config.non_sparse_matrix, sizeof(S) * mds_matrix_len, stream_copy_non_sparse);
    cudaMallocAsync(&this->config.sparse_matrices, sizeof(S) * sparse_matrices_len, stream_copy_sparse_matrices);

    // Copy constants
    cudaMemcpyAsync(
      this->config.round_constants, constants, sizeof(S) * round_constants_len, cudaMemcpyHostToDevice,
      stream_copy_round_constants);
    cudaMemcpyAsync(
      this->config.mds_matrix, mds_offset, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice, stream_copy_mds_matrix);
    cudaMemcpyAsync(
      this->config.non_sparse_matrix, non_sparse_offset, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice,
      stream_copy_non_sparse);
    cudaMemcpyAsync(
      this->config.sparse_matrices, sparse_matrices_offset, sizeof(S) * sparse_matrices_len, cudaMemcpyHostToDevice,
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

  ~Poseidon()
  {
    cudaFreeAsync(this->config.round_constants, this->stream);
    cudaFreeAsync(this->config.mds_matrix, this->stream);
    cudaFreeAsync(this->config.non_sparse_matrix, this->stream);
    cudaFreeAsync(this->config.sparse_matrices, this->stream);
  }

  // Hash multiple preimages in parallel
  cudaError_t hash_blocks(const S* inp, size_t blocks, S* out, HashType hash_type, cudaStream_t stream);

private:
  S tree_domain_tag, const_input_no_pad_domain_tag;
  cudaStream_t stream;
};