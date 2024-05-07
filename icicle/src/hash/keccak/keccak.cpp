#include "hash/keccak/keccak.cuh"

namespace keccak {
  template <int C, int D>
  cudaError_t
  keccak_hash(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config)
  {
    return 0;
  }

  extern "C" cudaError_t
  keccak256_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config)
  {
    return 0;
  }

  extern "C" cudaError_t
  keccak512_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config)
  {
    return 0;
  }
} // namespace keccak