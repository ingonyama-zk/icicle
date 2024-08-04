#include "utils/utils.h"

#include "fields/field_config.cuh"
using namespace field_config;

#include "gpu-utils/error_handler.cuh"
#include "poseidon2/poseidon2.cuh"
#include "./constants.cu"

namespace poseidon2 {
  template class Poseidon2<scalar_t>;

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_create_cuda)(
    Poseidon2<scalar_t>** poseidon,
    unsigned int width,
    unsigned int rate,
    unsigned int alpha,
    unsigned int internal_rounds,
    unsigned int external_rounds,
    const scalar_t* round_constants,
    const scalar_t* internal_matrix_diag,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx)
  {
    try {
      *poseidon = new Poseidon2<scalar_t>(
        width, rate, alpha, internal_rounds, external_rounds, round_constants, internal_matrix_diag, mds_type,
        diffusion, ctx);
      return cudaError_t::cudaSuccess;
    } catch (const IcicleError& _error) {
      return cudaError_t::cudaErrorUnknown;
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_load_cuda)(
    Poseidon2<scalar_t>** poseidon,
    unsigned int width,
    unsigned int rate,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx)
  {
    try {
      *poseidon = new Poseidon2<scalar_t>(width, rate, mds_type, diffusion, ctx);
      return cudaError_t::cudaSuccess;
    } catch (const IcicleError& _error) {
      return cudaError_t::cudaErrorUnknown;
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_hash_many_cuda)(
    const Poseidon2<scalar_t>* poseidon,
    const scalar_t* inputs,
    scalar_t* output,
    unsigned int number_of_states,
    unsigned int input_block_len,
    unsigned int output_len,
    hash::HashConfig& cfg)
  {
    return poseidon->hash_many(inputs, output, number_of_states, input_block_len, output_len, cfg);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_delete_cuda)(Poseidon2<scalar_t>* poseidon)
  {
    try {
      poseidon->~Poseidon2();
      return cudaError_t::cudaSuccess;
    } catch (const IcicleError& _error) {
      return cudaError_t::cudaErrorUnknown;
    }
  }
} // namespace poseidon2