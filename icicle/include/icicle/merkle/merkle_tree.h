#pragma once

#include <memory>
#include <array>
#include <vector>
#include "icicle/merkle/merkle_tree_config.h"
#include "icicle/merkle/merkle_proof.h"
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
     */
    explicit MerkleTree(std::shared_ptr<MerkleTreeBackend> backend) : m_backend{std::move(backend)} {}

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
    build(const T* leaves, uint64_t size /* number of leaf elements */, const MerkleTreeConfig& config)
    {
      return build(reinterpret_cast<const std::byte*>(leaves), size * sizeof(T), config);
    }

    /**
     * @brief Returns a pair containing the pointer to the root (ON HOST) data and its size.
     * @return A pair of (root data pointer, root size).
     */
    inline std::pair<std::byte*, size_t> get_merkle_root() const { return m_backend->get_merkle_root(); }

    template <typename T>
    inline std::pair<T*, size_t> get_merkle_root() const
    {
      auto [root, size] = get_merkle_root();
      return {reinterpret_cast<T*>(root), size / sizeof(T)};
    }

    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param leaf_idx Index of the element for which the Merkle path is required.
     * @param config Configuration for the Merkle tree operation.
     * @param merkle_proof Reference to the MerkleProof object where the path will be stored.
     * @return Error code of type eIcicleError.
     */
    inline eIcicleError get_merkle_proof(
      const std::byte* leaves,
      uint64_t leaf_idx,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof /*output*/) const
    {
      return m_backend->get_merkle_proof(leaves, leaf_idx, config, merkle_proof);
    }

    template <typename T>
    inline eIcicleError get_merkle_proof(
      const T* leaves, uint64_t leaf_idx, const MerkleTreeConfig& config, MerkleProof& merkle_proof /*output*/) const
    {
      return get_merkle_proof(reinterpret_cast<const std::byte*>(leaves), leaf_idx, config, merkle_proof);
    }

    /**
     * @brief Verify an element against the Merkle path using layer hashes.
     * @param merkle_proof The MerkleProof object includes the leaf, path, and the root.
     * @param valid output valid bit. True if the Proof is valid, false otherwise.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    eIcicleError verify(const MerkleProof& merkle_proof, bool& valid /*output*/) const
    {
      // TODO: Implement Merkle path verification here

      // can access path by offset or all of it
      // auto digest = merkle_proof.access_path_at_offset<DIGEST_TYPE>(offset);
      auto [path, path_size] = merkle_proof.get_path();
      auto [root, root_size] = merkle_proof.get_root();
      auto [leaf, leaf_size, leaf_idx] = merkle_proof.get_leaf();

      valid = true; // TODO use hashers to check path from leaf to root is recomputing the expected root
      return eIcicleError::SUCCESS;
    }

  private:
    std::shared_ptr<MerkleTreeBackend>
      m_backend; ///< Shared pointer to the backend responsible for Merkle tree operations.
  };

} // namespace icicle