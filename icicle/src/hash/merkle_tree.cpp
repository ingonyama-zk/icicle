#include "icicle/errors.h"
#include "icicle/hash/merkle_tree.h"
#include "icicle/backend/hash/merkle_tree_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(MerkleTreeDispatcher, merkle_tree_factory, MerkleTreeFactoryImpl);

  MerkleTree create_merkle_tree(
    const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size_in_limbs, uint64_t output_store_min_layer)
  {
    std::shared_ptr<MerkleTreeBackend> backend;
    ICICLE_CHECK(
      MerkleTreeDispatcher::execute(layer_hashes, leaf_element_size_in_limbs, output_store_min_layer, backend));
    MerkleTree merkle_tree{backend};
    return merkle_tree;
  }

  /*************************** C API ***************************/

  MerkleTree* create_merkle_tree_c_api(
    const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size_in_limbs, uint64_t output_store_min_layer)
  {
    return new MerkleTree(create_merkle_tree(layer_hashes, leaf_element_size_in_limbs, output_store_min_layer));
  }

} // namespace icicle