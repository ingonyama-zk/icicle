#include "fields/field_config.cuh"

using namespace field_config;

#include "poseidon/poseidon.cuh"
#include "constants.cu"
#include "extern_old.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace poseidon {
  typedef class Poseidon<scalar_t> PoseidonInst;

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_create_cuda)(
    PoseidonInst** poseidon,
    unsigned int arity,
    unsigned int alpha,
    unsigned int partial_rounds,
    unsigned int full_rounds_half,
    const scalar_t* round_constants,
    const scalar_t* mds_matrix,
    const scalar_t* non_sparse_matrix,
    const scalar_t* sparse_matrices,
    const scalar_t& domain_tag,
    device_context::DeviceContext& ctx)
  {
    try {
      *poseidon = new PoseidonInst(
        arity, alpha, partial_rounds, full_rounds_half, round_constants, mds_matrix, non_sparse_matrix, sparse_matrices,
        domain_tag, ctx);
      return cudaError_t::cudaSuccess;
    } catch (const IcicleError& _error) {
      return cudaError_t::cudaErrorUnknown;
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_load_cuda)(
    PoseidonInst** poseidon, unsigned int arity, device_context::DeviceContext& ctx)
  {
    try {
      *poseidon = new PoseidonInst(arity, ctx);
      return cudaError_t::cudaSuccess;
    } catch (const IcicleError& _error) {
      return cudaError_t::cudaErrorUnknown;
    }
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_absorb_many_cuda)(
    const PoseidonInst* poseidon,
    const scalar_t* inputs,
    scalar_t* states,
    unsigned int number_of_states,
    unsigned int input_block_len,
    const SpongeConfig& cfg)
  {
    return poseidon->absorb_many(inputs, states, number_of_states, input_block_len, cfg);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_squeeze_many_cuda)(
    const PoseidonInst* poseidon,
    const scalar_t* states,
    scalar_t* output,
    unsigned int number_of_states,
    unsigned int output_len,
    const SpongeConfig& cfg)
  {
    return poseidon->squeeze_many(states, output, number_of_states, output_len, cfg);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_hash_many_cuda)(
    const PoseidonInst* poseidon,
    const scalar_t* inputs,
    scalar_t* output,
    unsigned int number_of_states,
    unsigned int input_block_len,
    unsigned int output_len,
    const SpongeConfig& cfg)
  {
    return poseidon->hash_many(inputs, output, number_of_states, input_block_len, output_len, cfg);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_delete_cuda)(PoseidonInst* poseidon)
  {
    try {
      poseidon->~Poseidon();
      return cudaError_t::cudaSuccess;
    } catch (const IcicleError& _error) {
      return cudaError_t::cudaErrorUnknown;
    }
  }
} // namespace poseidon