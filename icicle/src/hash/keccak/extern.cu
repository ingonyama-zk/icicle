#include "utils/utils.h"

#include "gpu-utils/error_handler.cuh"

namespace keccak {
  extern "C" cudaError_t
  keccak256_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config)
  {
    return keccak_hash<512, 256>(input, input_block_size, number_of_blocks, output, config);
  }

  extern "C" cudaError_t
  keccak512_cuda(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config)
  {
    return keccak_hash<1024, 512>(input, input_block_size, number_of_blocks, output, config);
  }
} // namespace keccak