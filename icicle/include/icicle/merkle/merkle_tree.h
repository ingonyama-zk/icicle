#pragma once

#include <memory>
#include <array>
#include <vector>
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/merkle/merkle_tree_config.h"

namespace icicle {

  /**
   * @brief Class for performing Merkle tree operations.
   *
   * This class provides a high-level interface for building and managing Merkle trees. The underlying
   * logic for tree operations, such as building, retrieving paths, and verifying, is delegated to the
   * backend, which may be device-specific (e.g., CPU, GPU).
   */
  class MerkleTree
  {
  public:
    /**
     * @brief Constructor for the MerkleTree class.
     * @param backend Shared pointer to the backend responsible for Merkle tree operations.
     */
    MerkleTree(std::shared_ptr<MerkleTreeBackend> backend) : m_backend(std::move(backend)) {}

    /**
     * @brief Build the Merkle tree from the provided leaves.
     * @param leaves Pointer to the leaves of the tree (input data).
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    inline eIcicleError build(const std::byte* leaves, const MerkleTreeConfig& config)
    {
      return m_backend->build(leaves, config);
    }

    /**
     * @brief Retrieve the root of the Merkle tree.
     * @param root Output parameter for the Merkle root.
     * @return Error code of type eIcicleError.
     */
    inline eIcicleError get_root(const std::byte*& root) const { return m_backend->get_root(root); }

    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param element_idx Index of the element for which the Merkle path is required.
     * @param path Output parameter for the Merkle path.
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError.
     */
    inline eIcicleError
    get_path(const std::byte* leaves, uint64_t element_idx, std::byte* path, const MerkleTreeConfig& config) const
    {
      return m_backend->get_path(leaves, element_idx, path, config);
    }

    /**
     * @brief Verify an element against the Merkle path.
     * @param path Pointer to the Merkle path.
     * @param element_idx Index of the element being verified.
     * @param verification_valid Output parameter indicating if the verification succeeded.
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    inline eIcicleError verify(
      const std::byte* path, unsigned int element_idx, bool& verification_valid, const MerkleTreeConfig& config) const
    {
      return m_backend->verify(path, element_idx, verification_valid, config);
    }

  private:
    std::shared_ptr<MerkleTreeBackend>
      m_backend; ///< Shared pointer to the backend responsible for Merkle tree operations.
  };

  /**
   * @brief Create a MerkleTree with specified layer hashes and configurations.
   *
   * @param layer_hashes A vector of Hash objects representing the hashes of each tree layer.
   * @param leaf_element_size Size of each leaf element.
   * @param output_store_min_layer Minimum layer at which output is stored (default is 0).
   * @return A MerkleTree object initialized with the specified backend.
   */
  MerkleTree create_merkle_tree(
    const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0);

} // namespace icicle