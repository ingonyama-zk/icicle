#include "utils/utils.h"

#include "gpu-utils/error_handler.cuh"
#include "merkle-tree/merkle.cuh"
#include "merkle.cu"

#include "hash/hash.cuh"
#include "poseidon2/poseidon2.cuh"

#include "fields/field_config.cuh"
using namespace field_config;

namespace merkle_tree {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, build_poseidon2_merkle_tree)(
    const scalar_t* leaves_digests,
    scalar_t* digests,
    unsigned int height,
    unsigned int input_block_len,
    const poseidon2::Poseidon2<scalar_t>* poseidon_compression,
    const poseidon2::Poseidon2<scalar_t>* poseidon_bottom_layer,
    const TreeBuilderConfig& tree_config)
  {
    return build_merkle_tree<scalar_t, scalar_t>(
      leaves_digests, digests, height, input_block_len, *poseidon_compression, *poseidon_bottom_layer, tree_config);
  }
} // namespace merkle_tree