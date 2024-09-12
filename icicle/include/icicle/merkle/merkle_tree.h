#pragma once

#include <memory>
#include <array>
#include <vector>
#include "icicle/merkle/merkle_tree_config.h"
#include "icicle/merkle/merkle_path.h"
#include "icicle/backend/merkle/merkle_tree_backend.h"

namespace icicle {

  class MerkleTree;

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
     * @brief Static factory method to create a MerkleTree instance.
     *
     * @param layer_hashes A vector of Hash objects representing the hashes of each tree layer.
     * @param leaf_element_size Size of each leaf element.
     * @param output_store_min_layer Minimum layer at which output is stored (default is 0).
     * @return A MerkleTree object initialized with the specified backend.
     */
    static MerkleTree
    create(const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0)
    {
      return create_merkle_tree(layer_hashes, leaf_element_size, output_store_min_layer);
    }

    /**
     * @brief Constructor for the MerkleTree class.
     * @param backend Shared pointer to the backend responsible for Merkle tree operations.
     * @param layer_hashes A vector of Hash objects representing the hashes of each tree layer.
     */
    MerkleTree(std::shared_ptr<MerkleTreeBackend> backend) : m_backend{std::move(backend)} {}

    /**
     * @brief Build the Merkle tree from the provided leaves.
     * @param leaves Pointer to the leaves of the tree (input data).
     * @param size The size of the leaves.
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    inline eIcicleError build(const std::byte* leaves, uint64_t size, const MerkleTreeConfig& config)
    {
      return m_backend->build(leaves, size, config);
    }

    template <typename T>
    inline eIcicleError
    build(const T* leaves, uint64_t size /* =number-of-leave-elements*/, const MerkleTreeConfig& config)
    {
      return build((const std::byte*)leaves, size, config);
    }

    /**
     * @brief Retrieve the root of the Merkle tree.
     * @param root Pointer to where the Merkle root will be written. Must be on host memory.
     * @return Error code of type eIcicleError.
     */
    inline eIcicleError get_merkle_root(std::byte* root /*output*/) const { return m_backend->get_merkle_root(root); }

    template <typename T>
    inline eIcicleError get_merkle_root(T* root /*output*/) const
    {
      return get_merkle_root((std::byte*)root);
    }

    /**
     * @brief Calculate the size of the Merkle path for an element.
     *
     * This function calculates the size of the Merkle path based on the number of levels
     * in the tree and the size of each hash.
     * @param pruned A pruned path the siblings required for computation, excluding the elements that can be computed
     * directly.
     *
     * @return The total size of the Merkle path in bytes.
     */
    inline uint64_t calculate_merkle_path_size(bool pruned = false) const
    {
      ICICLE_ASSERT(!pruned) << "TODO support pruned merkle paths";
      // TODO: pruned paths can be smaller
      uint64_t merkle_path_size = 0;
      for (const auto& layer : m_backend->get_layer_hashes()) {
        merkle_path_size += layer.output_size();
      }
      return merkle_path_size;
    }

    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param element_idx Index of the element for which the Merkle path is required.
     * @param config Configuration for the Merkle tree operation.
     * @param merkle_path Reference to the MerklePath object where the path will be stored.
     * @return Error code of type eIcicleError.
     */
    inline eIcicleError get_merkle_path(
      const std::byte* leaves,
      uint64_t element_idx,
      const MerkleTreeConfig& config,
      MerklePath& merkle_path /*output*/) const
    {
      // Ask backend to generate the path and store it in the MerklePath object
      return m_backend->get_merkle_path(leaves, element_idx, config, merkle_path);
    }

    template <typename T>
    inline eIcicleError get_merkle_path(
      const T* leaves, uint64_t element_idx, const MerkleTreeConfig& config, MerklePath& merkle_path /*output*/) const
    {
      // Ask backend to generate the path and store it in the MerklePath object
      return get_merkle_path((std::byte*)leaves, element_idx, config, merkle_path);
    }

    /**
     * @brief Verify an element against the Merkle path using layer hashes (frontend verification).
     * @param element Pointer to the element being verified.
     * @param element_idx Index of the element being verified.
     * @param merkle_path The MerklePath object representing the path.
     * @param root Pointer to the root of the Merkle tree.
     * @param config Configuration for the Merkle tree operation.
     * @return True if the verification succeeds, false otherwise.
     */
    bool verify(
      const std::byte* element,
      uint64_t element_idx,
      const MerklePath& merkle_path,
      const std::byte* root,
      const MerkleTreeConfig& config) const
    {
      // TODO implement merkle-path verification
      return true;
    }

    template <typename T, typename R>
    bool verify(
      const T* element,
      uint64_t element_idx,
      const MerklePath& merkle_path,
      const R* root,
      const MerkleTreeConfig& config) const
    {
      return verify((const std::byte*)element, element_idx, merkle_path, (const std::byte*)root, config);
    }

  private:
    std::shared_ptr<MerkleTreeBackend>
      m_backend; ///< Shared pointer to the backend responsible for Merkle tree operations.
  };

} // namespace icicle