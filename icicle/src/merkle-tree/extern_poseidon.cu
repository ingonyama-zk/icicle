#include "utils/utils.h"

#include "gpu-utils/error_handler.cuh"
#include "merkle-tree/merkle.cuh"
#include "merkle.cu"

#include "hash/hash.cuh"
#include "poseidon/poseidon.cuh"

#include "fields/field_config.cuh"
using namespace field_config;

namespace merkle_tree {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, build_poseidon_merkle_tree)(
    const scalar_t* leaves_digests,
    scalar_t* digests,
    unsigned int height,
    unsigned int arity,
    unsigned int input_block_len, 
    const poseidon::Poseidon<scalar_t>* poseidon_compression,
    const poseidon::Poseidon<scalar_t>* poseidon_sponge,
    const hash::SpongeConfig& sponge_config,
    const TreeBuilderConfig& tree_config)
  {
    return build_merkle_tree<poseidon::Poseidon<scalar_t>, scalar_t, scalar_t>(
      leaves_digests,
      digests,
      height,
      arity,
      input_block_len,
      *poseidon_compression,
      *poseidon_sponge,
      sponge_config,
      tree_config
    );
  }
} // namespace merkle_tree