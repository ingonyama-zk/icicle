#define FIELD_ID BN254
#include "../../include/fields/field_config.cuh"

using namespace field_config;

#include "../../include/poseidon/poseidon.cuh"
#include "constants.cu"
#include "kernels.cu"
typedef int cudaError_t;
namespace poseidon {
  template <typename S, int T>
  cudaError_t
  permute_many(S* states, size_t number_of_states, const PoseidonConstants<S>& constants, int& stream)
  {
    return 0;
  }

  template <typename S, int T>
  cudaError_t poseidon_hash(
    S* input, S* output, size_t number_of_states, const PoseidonConstants<S>& constants, const PoseidonConfig& config)
  {
    return 0;
  }

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
    return 0;
  }
} // namespace poseidon