#include "utils/utils.h"

#include "fields/field_config.cuh"
using namespace field_config;

#include "poseidon.cu"

namespace poseidon2 {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, create_optimized_poseidon2_constants_cuda)(
    int width,
    int alpha,
    int internal_rounds,
    int external_rounds,
    const scalar_t* round_constants,
    const scalar_t* internal_matrix_diag,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<scalar_t>* poseidon_constants)
  {
    return create_optimized_poseidon2_constants<scalar_t>(
      width, alpha, internal_rounds, external_rounds, round_constants, internal_matrix_diag, mds_type, diffusion, ctx,
      poseidon_constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, init_optimized_poseidon2_constants_cuda)(
    int width,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<scalar_t>* constants)
  {
    return init_optimized_poseidon2_constants<scalar_t>(width, mds_type, diffusion, ctx, constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_hash_cuda)(
    scalar_t* input,
    scalar_t* output,
    int number_of_states,
    int width,
    const Poseidon2Constants<scalar_t>* constants,
    Poseidon2Config* config)
  {
#define P2_HASH_T(width)                                                                                               \
  case width:                                                                                                          \
    return poseidon2_hash<scalar_t, width>(input, output, number_of_states, *constants, *config);

    switch (width) {
      P2_HASH_T(2)
      P2_HASH_T(3)
      P2_HASH_T(4)
      P2_HASH_T(8)
      P2_HASH_T(12)
      P2_HASH_T(16)
      P2_HASH_T(20)
      P2_HASH_T(24)
    default:
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument, "PoseidonHash: #arity must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
    }
    return CHK_LAST();
  }
} // namespace poseidon2