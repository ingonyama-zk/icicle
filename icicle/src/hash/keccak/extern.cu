#include "utils/utils.h"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"
#include "hash/keccak/keccak.cuh"
#include "keccak.cu"

namespace keccak {
  extern "C" cudaError_t
  keccak256_cuda(uint8_t* input, int input_block_size, int number_of_states, uint8_t* output, KeccakConfig& config)
  {
    SpongeConfig cfg{
      config.are_inputs_on_device,
      config.are_outputs_on_device,
      input_block_size,
      32,
      136,
      17,
      0,
      config.ctx,
      config.is_async};
    return Keccak().hash_many(input, (uint64_t*)output, number_of_states, cfg);
  }

  extern "C" cudaError_t
  keccak512_cuda(uint8_t* input, int input_block_size, int number_of_states, uint8_t* output, KeccakConfig& config)
  {
    SpongeConfig cfg{config.are_inputs_on_device,
                     config.are_outputs_on_device,
                     input_block_size,
                     64,
                     72,
                     9,
                     0,
                     config.ctx,
                     config.is_async};
    return Keccak().hash_many(input, (uint64_t*)output, number_of_states, cfg);
  }
} // namespace keccak