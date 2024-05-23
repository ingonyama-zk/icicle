#include "utils/utils.h"

#include "fields/field_config.cuh"
using namespace field_config;

#include "gpu-utils/error_handler.cuh"
#include "merkle-tree/merkle.cuh"

#include "poseidon2/poseidon2.cuh"

namespace merkle_tree {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, build_poseidon2_merkle_tree)(
    const scalar_t* leaves_digests,
    scalar_t* digests,
    uint32_t height,
    int arity,
    Poseidon2<scalar_t>& poseidon,
    TreeBuilderConfig& config)
  {
    return build_merkle_tree<scalar_t>(leaves_digests, digests, height, poseidon, config);
  }
} // namespace merkle_tree