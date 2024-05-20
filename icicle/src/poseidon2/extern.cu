#include "utils/utils.h"

#include "fields/field_config.cuh"
using namespace field_config;

#include "poseidon2/poseidon2.cuh"
#include "poseidon2.cu"

namespace poseidon2 {
  template class Poseidon2<scalar_t>;

  template void poseidon2_permutation_kernel<scalar_t, 3>(
    const scalar_t* states, scalar_t* states_out, unsigned int number_of_states, const Poseidon2Constants<scalar_t> constants);

  template void internal_round<scalar_t, 3>(scalar_t state[3], size_t rc_offset, const Poseidon2Constants<scalar_t>& constants);

  template void add_rc<scalar_t, 3>(scalar_t state[3], size_t rc_offset, const scalar_t* rc);

  template void sbox<scalar_t, 3>(scalar_t state[3], const int alpha);

  template void mds_light<scalar_t, 3>(scalar_t state[3], MdsType mds);

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, create_poseidon2_constants_cuda)(
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
    return create_poseidon2_constants<scalar_t>(
      width, alpha, internal_rounds, external_rounds, round_constants, internal_matrix_diag, mds_type, diffusion, ctx,
      poseidon_constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, init_poseidon2_constants_cuda)(
    int width,
    MdsType mds_type,
    DiffusionStrategy diffusion,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<scalar_t>* constants)
  {
    return init_poseidon2_constants<scalar_t>(width, mds_type, diffusion, ctx, constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_permute_many_cuda)(
    const scalar_t* states,
    scalar_t* output,
    int number_of_states,
    Poseidon2<scalar_t>* poseidon,
    device_context::DeviceContext& ctx
  )
  {
    return poseidon->permute_many(states, output, number_of_states, ctx);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon2_compress_many_cuda)(
    const scalar_t* states,
    scalar_t* output,
    int number_of_states,
    Poseidon2<scalar_t>* poseidon,
    device_context::DeviceContext& ctx,
    scalar_t* perm_output
  )
  {
    return poseidon->compress_many(states, output, number_of_states, 1, ctx, 0, perm_output);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, release_poseidon2_constants_cuda)(
    Poseidon2Constants<scalar_t>* constants, device_context::DeviceContext& ctx)
  {
    return release_poseidon2_constants<scalar_t>(constants, ctx);
  }
} // namespace poseidon2