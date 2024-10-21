#include "icicle/errors.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(MerkleTreeDispatcher, merkle_tree_factory, MerkleTreeFactoryImpl);

  MerkleTree
  create_merkle_tree(const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer)
  {
    std::shared_ptr<MerkleTreeBackend> backend;
    ICICLE_CHECK(MerkleTreeDispatcher::execute(layer_hashes, leaf_element_size, output_store_min_layer, backend));
    MerkleTree merkle_tree{backend};
    return merkle_tree;
  }

} // namespace icicle