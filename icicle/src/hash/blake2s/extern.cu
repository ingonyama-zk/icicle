#include "utils/utils.h"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"

#include "hash/blake2s/blake2s.cuh"
#include "blake2s.cu"
#include "../../merkle-tree/merkle.cu"
#include "../../merkle-tree/mmcs.cu"
#include "merkle-tree/merkle.cuh"
#include "matrix/matrix.cuh"

namespace blake2s {
  extern "C" cudaError_t blake2s_cuda(
    BYTE* input, BYTE* output, WORD number_of_blocks, WORD input_block_size, WORD output_block_size, HashConfig& config)
  {
    return Blake2s().hash_many(input, output, number_of_blocks, input_block_size, output_block_size, config);
  }

  extern "C" cudaError_t build_blake2s_merkle_tree_cuda(
    const BYTE* leaves,
    BYTE* digests,
    unsigned int height,
    WORD input_block_len,
    const merkle_tree::TreeBuilderConfig& tree_config)
  {
    Blake2s blake2s;
    return merkle_tree::build_merkle_tree<BYTE, BYTE>(
      leaves, digests, height, input_block_len, blake2s, blake2s, tree_config);
  }

  extern "C" cudaError_t blake2s_mmcs_commit_cuda(
    const Matrix<BYTE>* leaves,
    unsigned int number_of_inputs,
    BYTE* digests,
    const merkle_tree::TreeBuilderConfig& tree_config)
  {
    Blake2s hasher;
    return merkle_tree::mmcs_commit<BYTE, BYTE>(leaves, number_of_inputs, digests, hasher, hasher, tree_config);
  }
} // namespace blake2s