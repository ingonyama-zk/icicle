#pragma once

#include <memory>
#include "icicle/common.h" // For limb and stream types
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/merkle/merkle_tree_config.h"

namespace icicle {

  class MerkleTree
  {
  public:
    /**
     * @brief Constructor for the MerkleTree class.
     * @param backend Shared pointer to the backend for Merkle tree operations.
     */
    MerkleTree(std::shared_ptr<MerkleTreeBackend> backend) : m_backend(std::move(backend)) {}

    /**
     * @brief Build the Merkle tree from the leaves.
     * @param leaves Pointer to the leaves of the tree.
     * @param config Configuration struct for the merkle tree.
     * @param secondary_leaves Pointer to the secondary_leaves in case the tree receives inputs from a secondary stream.
     * @return Error code of type eIcicleError.
     */
    eIcicleError build(const limb_t* leaves, const MerkleTreeConfig& config, const limb_t* secondary_leaves = nullptr)
    {
      return m_backend->build(leaves, config, secondary_leaves);
    }

    /**
     * @brief Get the root of the Merkle tree.
     * @param root Output parameter for the root of the tree.
     * @return Error code of type eIcicleError.
     */
    eIcicleError get_root(const limb_t*& root) const { return m_backend->get_root(root); }

    /**
     * @brief Get the Merkle path from a specified element index.
     * @param leaves Pointer to the leaves of the tree.
     * @param element_idx Index of the element for which the path is to be retrieved.
     * @param path Output parameter for the Merkle path.
     * @param config Configuration struct for the Merkle tree.
     * @return Error code of type eIcicleError.
     */
    eIcicleError
    get_path(const limb_t* leaves, uint64_t element_idx, limb_t* path, const MerkleTreeConfig& config) const
    {
      return m_backend->get_path(leaves, element_idx, path, config);
    }

    /**
     * @brief Verify an element against a Merkle path.
     * @param path Pointer to the Merkle path.
     * @param element_idx Index of the element to verify.
     * @param verification_valid Output parameter indicating if the verification is valid.
     * @param config Configuration struct for the Merkle tree.
     * @return Error code of type eIcicleError.
     */
    eIcicleError
    verify(const limb_t* path, unsigned int element_idx, bool& verification_valid, const MerkleTreeConfig& config)
    {
      return m_backend->verify(path, element_idx, verification_valid, config);
    }

  private:
    std::shared_ptr<MerkleTreeBackend> m_backend; ///< Shared pointer to the Merkle tree backend.
  };

  MerkleTree create_merkle_tree(
    const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size_in_limbs, uint64_t output_store_min_layer);

} // namespace icicle