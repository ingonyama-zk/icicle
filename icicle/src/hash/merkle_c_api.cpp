#include "icicle/utils/log.h"
#include "icicle/errors.h"
#include "icicle/merkle/merkle_proof.h"

extern "C" {
// Define an opaque pointer type for MerkleProof (similar to a handle in C)
typedef icicle::MerkleProof* MerkleProofHandle;

// Create a new MerkleProof object and return a handle to it.
MerkleProofHandle icicle_merkle_proof_create() { return new icicle::MerkleProof(); }

// Delete the MerkleProof object and free its resources.
eIcicleError icicle_merkle_proof_delete(MerkleProofHandle proof)
{
  if (!proof) { return eIcicleError::INVALID_POINTER; }
  delete proof;
  return eIcicleError::SUCCESS;
}

// Allocate memory for the Merkle proof and copy the leaf and root data.
eIcicleError icicle_merkle_proof_allocate(
  MerkleProofHandle proof,
  bool pruned_path,
  uint64_t leaf_idx,
  const std::byte* leaf,
  std::size_t leaf_size,
  const std::byte* root,
  std::size_t root_size)
{
  if (!proof) { return eIcicleError::INVALID_POINTER; }
  try {
    proof->allocate(pruned_path, leaf_idx, leaf, leaf_size, root, root_size);
  } catch (const std::exception& e) {
    ICICLE_LOG_ERROR << "Exception in icicle_merkle_proof_allocate: " << e.what();
    return eIcicleError::ALLOCATION_FAILED;
  }
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

// Push a node to the Merkle path using raw byte data.
eIcicleError icicle_merkle_proof_push_node(MerkleProofHandle proof, const std::byte* node, std::size_t size)
{
  if (!proof || !node) { return eIcicleError::INVALID_POINTER; }
  try {
    proof->push_node_to_path(node, size);
  } catch (const std::exception& e) {
    ICICLE_LOG_ERROR << "Exception in icicle_merkle_proof_push_node: " << e.what();
    return eIcicleError::INVALID_POINTER;
  }
  return eIcicleError::SUCCESS;
}

} // extern "C"