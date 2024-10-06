#include "icicle/utils/log.h"
#include "icicle/errors.h"
#include "icicle/merkle/merkle_proof.h"
#include "icicle/merkle/merkle_tree.h"

extern "C" {
// Define an opaque pointer type for MerkleProof (similar to a handle in C)
typedef icicle::MerkleProof* MerkleProofHandle;
typedef icicle::MerkleTree* MerkleTreeHandle;

// Create a new MerkleProof object and return a handle to it.
MerkleProofHandle icicle_merkle_proof_create() { return new icicle::MerkleProof(); }

// Delete the MerkleProof object and free its resources.
eIcicleError icicle_merkle_proof_delete(MerkleProofHandle proof)
{
  if (!proof) { return eIcicleError::INVALID_POINTER; }
  delete proof;
  return eIcicleError::SUCCESS;
}

// Check if the Merkle path is pruned.
bool icicle_merkle_proof_is_pruned(MerkleProofHandle proof)
{
  if (!proof) { return false; }
  return proof->is_pruned();
}

// Get the path data and its size.
const std::byte* icicle_merkle_proof_get_path(MerkleProofHandle proof, std::size_t* out_size)
{
  if (!proof || !out_size) { return nullptr; }
  auto [path, size] = proof->get_path();
  *out_size = size;
  return path;
}

// Get the leaf data, its size, and its index.
const std::byte* icicle_merkle_proof_get_leaf(MerkleProofHandle proof, std::size_t* out_size, uint64_t* out_leaf_idx)
{
  if (!proof || !out_size || !out_leaf_idx) { return nullptr; }
  auto [leaf, size, leaf_idx] = proof->get_leaf();
  *out_size = size;
  *out_leaf_idx = leaf_idx;
  return leaf;
}

// Get the root data and its size.
const std::byte* icicle_merkle_proof_get_root(MerkleProofHandle proof, std::size_t* out_size)
{
  if (!proof || !out_size) { return nullptr; }
  auto [root, size] = proof->get_root();
  *out_size = size;
  return root;
}

// Create a new MerkleTree object
icicle::MerkleTree* icicle_merkle_tree_create(
  const icicle::Hash** layer_hashes,
  size_t layer_hashes_len,
  uint64_t leaf_element_size,
  uint64_t output_store_min_layer)
{
  try {
    // Convert the array of Hash pointers to a vector of Hash objects
    std::vector<icicle::Hash> hash_vector;
    hash_vector.reserve(layer_hashes_len);

    // Dereference each pointer and push the Hash object into the vector
    for (size_t i = 0; i < layer_hashes_len; ++i) {
      if (layer_hashes[i] == nullptr) {
        // Handle the case where one of the pointers is null, if needed
        throw std::invalid_argument("Null pointer found in layer_hashes.");
      }
      hash_vector.push_back(*layer_hashes[i]);
    }

    // Create a MerkleTree instance using the static factory method
    return new icicle::MerkleTree(icicle::MerkleTree::create(hash_vector, leaf_element_size, output_store_min_layer));
  } catch (...) {
    return nullptr;
  }
}

// Delete the MerkleTree object and free its resources
void icicle_merkle_tree_delete(icicle::MerkleTree* tree)
{
  if (tree) { delete tree; }
}

// Build the Merkle tree from the provided leaves
eIcicleError icicle_merkle_tree_build(
  icicle::MerkleTree* tree, const std::byte* leaves, uint64_t size, const icicle::MerkleTreeConfig* config)
{
  if (!tree || !leaves || !config) { return eIcicleError::INVALID_POINTER; }

  try {
    return tree->build(leaves, size, *config);
  } catch (...) {
    return eIcicleError::UNKNOWN_ERROR;
  }
}

// Get the Merkle root as a pointer to the root data and its size
const std::byte* icicle_merkle_tree_get_root(icicle::MerkleTree* tree, size_t* out_size)
{
  if (!tree || !out_size) { return nullptr; }

  try {
    auto [root_ptr, root_size] = tree->get_merkle_root();
    *out_size = root_size;
    return root_ptr;
  } catch (...) {
    return nullptr;
  }
}

// Retrieve the Merkle proof for a specific element
eIcicleError icicle_merkle_tree_get_proof(
  icicle::MerkleTree* tree,
  const std::byte* leaves,
  uint64_t leaf_idx,
  const icicle::MerkleTreeConfig* config,
  icicle::MerkleProof* merkle_proof)
{
  if (!tree || !leaves || !config || !merkle_proof) { return eIcicleError::INVALID_POINTER; }

  try {
    return tree->get_merkle_proof(leaves, leaf_idx, *config, *merkle_proof);
  } catch (...) {
    return eIcicleError::UNKNOWN_ERROR;
  }
}

// Verify a Merkle proof
eIcicleError icicle_merkle_tree_verify(icicle::MerkleTree* tree, const icicle::MerkleProof* merkle_proof, bool* valid)
{
  if (!tree || !merkle_proof || !valid) { return eIcicleError::INVALID_POINTER; }

  try {
    return tree->verify(*merkle_proof, *valid);
  } catch (...) {
    return eIcicleError::UNKNOWN_ERROR;
  }
}

} // extern "C"