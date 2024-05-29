#include "utils/utils.h"

#include "gpu-utils/error_handler.cuh"
#include "merkle-tree/merkle.cuh"

#include "hash/hash.cuh"
#include "hash/keccak.cuh"

namespace merkle_tree {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, build_keccak512_merkle_tree)(
    const scalar_t* leaves, scalar_t* digests, unsigned int height, Keccak<512>& keccak, TreeBuilderConfig& config)
  {
    return build_merkle_tree<scalar_t>(leaves_digests, digests, height, keccak, config);
  }
} // namespace merkle_tree