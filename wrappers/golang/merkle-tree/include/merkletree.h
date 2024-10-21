#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifndef _MERKLETREE
  #define _MERKLETREE

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct MerkleTree MerkleTree;
typedef struct MerkleTreeConfig MerkleTreeConfig;
typedef struct MerkleProof MerkleProof;
typedef struct Hash Hash;

MerkleProof* icicle_merkle_proof_create();

// Delete the MerkleProof object and free its resources.
int icicle_merkle_proof_delete(MerkleProof* proof);

// Check if the Merkle path is pruned.
bool icicle_merkle_proof_is_pruned(MerkleProof* proof);

// Get the path data and its size.
uint8_t* icicle_merkle_proof_get_path(MerkleProof* proof, size_t* out_size);

// Get the leaf data, its size, and its index.
uint8_t* icicle_merkle_proof_get_leaf(MerkleProof* proof, size_t* out_size, uint64_t* out_leaf_idx);

// Get the root data and its size.
uint8_t* icicle_merkle_proof_get_root(MerkleProof* proof, size_t* out_size);

// Create a new MerkleTree object
MerkleTree* icicle_merkle_tree_create(
  Hash** layer_hashes,
  size_t layer_hashes_len,
  uint64_t leaf_element_size,
  uint64_t output_store_min_layer);

// Delete the MerkleTree object and free its resources
int icicle_merkle_tree_delete(MerkleTree* tree);

// Build the Merkle tree from the provided leaves
int icicle_merkle_tree_build(
  MerkleTree* tree, uint8_t* leaves, uint64_t size, MerkleTreeConfig* config);

// Get the Merkle root as a pointer to the root data and its size
const uint8_t* icicle_merkle_tree_get_root(MerkleTree* tree, size_t* out_size);

// Retrieve the Merkle proof for a specific element
int icicle_merkle_tree_get_proof(
  MerkleTree* tree,
  uint8_t* leaves,
  uint64_t leaves_size,
  uint64_t leaf_idx,
  bool is_pruned,
  MerkleTreeConfig* config,
  MerkleProof* merkle_proof);

// Verify a Merkle proof
int icicle_merkle_tree_verify(MerkleTree* tree, MerkleProof* merkle_proof, bool* valid);


  #ifdef __cplusplus
}
  #endif

#endif