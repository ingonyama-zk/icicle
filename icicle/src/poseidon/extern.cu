#include "fields/field_config.cuh"

using namespace field_config;

#include "poseidon.cu"
#include "constants.cu"


#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace poseidon {
  /**
   * Extern "C" version of [poseidon_hash_cuda] function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_hash_cuda)(
    scalar_t* input,
    scalar_t* output,
    int number_of_states,
    int arity,
    const PoseidonConstants<scalar_t>& constants,
    PoseidonConfig& config)
  {
    switch (arity) {
    case 2:
      return poseidon_hash<scalar_t, 3>(input, output, number_of_states, constants, config);
    case 4:
      return poseidon_hash<scalar_t, 5>(input, output, number_of_states, constants, config);
    case 8:
      return poseidon_hash<scalar_t, 9>(input, output, number_of_states, constants, config);
    case 11:
      return poseidon_hash<scalar_t, 12>(input, output, number_of_states, constants, config);
    default:
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "PoseidonHash: #arity must be one of [2, 4, 8, 11]");
    }
    return CHK_LAST();
  }

    extern "C" cudaError_t CONCAT_EXPAND(FIELD, create_optimized_poseidon_constants_cuda)(
    int arity,
    int full_rounds_half,
    int partial_rounds,
    const scalar_t* constants,
    device_context::DeviceContext& ctx,
    PoseidonConstants<scalar_t>* poseidon_constants)
  {
    return create_optimized_poseidon_constants<scalar_t>(
      arity, full_rounds_half, partial_rounds, constants, ctx, poseidon_constants);
  }

    extern "C" cudaError_t CONCAT_EXPAND(FIELD, init_optimized_poseidon_constants_cuda)(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<scalar_t>* constants)
  {
    return init_optimized_poseidon_constants<scalar_t>(arity, ctx, constants);
  }
} // namespace poseidon