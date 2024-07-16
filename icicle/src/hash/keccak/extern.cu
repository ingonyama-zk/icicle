#include "utils/utils.h"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"
#include "hash/keccak/keccak.cuh"
#include "keccak.cu"
#include "../../merkle-tree/merkle.cu"
#include "merkle-tree/merkle.cuh"

namespace keccak {
  extern "C" cudaError_t
  keccak256_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, HashConfig& config)
  {
    return Keccak(KECCAK_256_RATE)
      .hash_many(input, (uint64_t*)output, number_of_blocks, input_block_size, KECCAK_256_DIGEST, config);
  }

  extern "C" cudaError_t
  keccak512_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, HashConfig& config)
  {
    return Keccak(KECCAK_512_RATE)
      .hash_many(input, (uint64_t*)output, number_of_blocks, input_block_size, KECCAK_512_DIGEST, config);
  }

  extern "C" cudaError_t build_keccak256_merkle_tree_cuda(
    const uint8_t* leaves,
    uint64_t* digests,
    unsigned int height,
    unsigned int input_block_len,
    const merkle_tree::TreeBuilderConfig& tree_config)
  {
    Keccak keccak(KECCAK_256_RATE);
    return merkle_tree::build_merkle_tree<uint8_t, uint64_t>(
      leaves, digests, height, input_block_len, keccak, keccak, tree_config);
  }

  extern "C" cudaError_t build_keccak512_merkle_tree_cuda(
    const uint8_t* leaves,
    uint64_t* digests,
    unsigned int height,
    unsigned int input_block_len,
    const merkle_tree::TreeBuilderConfig& tree_config)
  {
    Keccak keccak(KECCAK_512_RATE);
    return merkle_tree::build_merkle_tree<uint8_t, uint64_t>(
      leaves, digests, height, input_block_len, keccak, keccak, tree_config);
  }

} // namespace keccak