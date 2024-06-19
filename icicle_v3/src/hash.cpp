#include "icicle/hash.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*********************************** Merkle tree ***********************************/
  ICICLE_DISPATCHER_INST(MerkleTreeDispatcher, merkle_tree, MerkleTreeImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, merkle_tree)(
    MerkleTree** merkle_tree, int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer,
               TreeBuilderConfig tree_config)
  {
    return MerkleTreeDispatcher::execute(merkle_tree, nof_layers, layer_hashes, output_store_min_layer, output_store_max_layer, tree_config);
  }

  
  eIcicleError merkle_tree(MerkleTree** merkle_tree, int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer,
               TreeBuilderConfig tree_config)
  {
    return CONCAT_EXPAND(FIELD, merkle_tree)(merkle_tree, nof_layers, layer_hashes, output_store_min_layer, output_store_max_layer, tree_config);
  }

  /*********************************** Poseidon Hash ***********************************/
    ICICLE_DISPATCHER_INST(PoseidonDispatcher, poseidon, PoseidonImpl);
    extern "C" eIcicleError CONCAT_EXPAND(FIELD, poseidon)(
      Hash** hash, int element_nof_limbs, int input_nof_elements, int output_nof_elements)
    {
      return PoseidonDispatcher::execute(hash, element_nof_limbs, input_nof_elements, output_nof_elements);
    }


    eIcicleError poseidon(Hash** hash, int element_nof_limbs, int input_nof_elements, int output_nof_elements)
  {
    return CONCAT_EXPAND(FIELD, poseidon)(hash, element_nof_limbs, input_nof_elements, output_nof_elements);
  }
}