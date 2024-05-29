#include "utils/utils.h"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"
#include "hash/keccak/keccak.cuh"
#include "keccak.cu"

namespace keccak {
  extern "C" cudaError_t keccak256_cuda(
    uint8_t* input, unsigned int input_block_size, unsigned int number_of_states, uint8_t* output, KeccakConfig& config)
  {
    SpongeConfig cfg{
      config.ctx,     config.are_inputs_on_device, config.are_outputs_on_device, 136, 17, 0,
      config.is_async};
    return Keccak().hash_many(input, (uint64_t*)output, number_of_states, input_block_size, 4, cfg);
  }

  extern "C" cudaError_t keccak512_cuda(
    uint8_t* input, unsigned int input_block_size, unsigned int number_of_states, uint8_t* output, KeccakConfig& config)
  {
    SpongeConfig cfg{
      config.ctx,     config.are_inputs_on_device, config.are_outputs_on_device, 72, 9, 0,
      config.is_async};
    return Keccak().hash_many(input, (uint64_t*)output, number_of_states, input_block_size, 4, cfg);
  }
} // namespace keccak