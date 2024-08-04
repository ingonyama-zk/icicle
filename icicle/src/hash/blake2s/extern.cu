#include "utils/utils.h"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"

#include "hash/blake2s/blake2s.cuh"
#include "blake2s.cu"
#include "../../merkle-tree/merkle.cu"
#include "merkle-tree/merkle.cuh"

namespace blake2s {
  extern "C" cudaError_t
  blake2s_cuda(BYTE * input, BYTE * output,  WORD number_of_blocks, WORD input_block_size, WORD output_block_size, HashConfig& config)
  {
    return Blake2s().hash_many(
      input, output, number_of_blocks, input_block_size, output_block_size, config);
  }

} // namespace blake2s