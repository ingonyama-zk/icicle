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
     * @param leaves_size The size of the leaves in bytes.
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    inline eIcicleError build(const std::byte* leaves, uint64_t leaves_size, const MerkleTreeConfig& config)
    {
      return m_backend->build(leaves, leaves_size, config);
    }

    /**
     * @brief Build the Merkle tree from the provided leaves.
     * @param leaves Pointer to the leaves of the tree (input data).
     * @param nof_leaves Number of T elements in leaves
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    template <typename T>
    inline eIcicleError
    build(const T* leaves, uint64_t nof_leaves /* number of leaf elements */, const MerkleTreeConfig& config)
    {
      return build(reinterpret_cast<const std::byte*>(leaves), nof_leaves * sizeof(T), config);
    }

    /**
     * @brief Returns a pair containing the pointer to the root (ON HOST) data and its size.
     * @return A pair of (root data pointer, root size).
     */
    inline std::pair<const std::byte*, size_t> get_merkle_root() const { return m_backend->get_merkle_root(); }

    template <typename T>
    inline std::pair<T*, size_t> get_merkle_root() const
    {
      auto [root, size] = get_merkle_root();
      return {reinterpret_cast<T*>(root), size / sizeof(T)};
    }

    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param leaves_size The size of the leaves in bytes.
     * @param leaf_idx Index of the element for which the Merkle path is required.
     * @param is_pruned If set, the path will not include hash results that can be extracted from siblings
     * @param config Configuration for the Merkle tree operation.
     * @param merkle_proof Reference to the MerkleProof object where the path will be stored.
     * @return Error code of type eIcicleError.
     */
    inline eIcicleError get_merkle_proof(
      const std::byte* leaves,
      uint64_t leaves_size,
      uint64_t leaf_idx,
      bool is_pruned,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof /*output*/) const
    {
      return m_backend->get_merkle_proof(leaves, leaves_size, leaf_idx, is_pruned, config, merkle_proof);
    }
    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param nof_leaves Number of T elements in leaves
     * @param leaf_idx Index of the element for which the Merkle path is required.
     * @param is_pruned If set, the path will not include hash results that can be extracted from siblings
     * @param config Configuration for the Merkle tree operation.
     * @param merkle_proof Reference to the MerkleProof object where the path will be stored.
     * @return Error code of type eIcicleError.
     */
    template <typename T>
    inline eIcicleError get_merkle_proof(
      const T* leaves,
      uint64_t nof_leaves,
      uint64_t leaf_idx,
      bool is_pruned,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof /*output*/) const
    {
      return get_merkle_proof(
        reinterpret_cast<const std::byte*>(leaves), nof_leaves * sizeof(T), leaf_idx, is_pruned, config, merkle_proof);
    }

    /**
     * @brief Verify an element against the Merkle path using layer hashes.
     * @param merkle_proof The MerkleProof object includes the leaf, path, and the root.
     * @param valid output valid bit. True if the Proof is valid, false otherwise.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    eIcicleError verify(const MerkleProof& merkle_proof, bool& valid /*output*/) const
    {
      const auto& layer_hashes = m_backend->get_layer_hashes();
      const uint nof_layers = layer_hashes.size();
      auto [leaf, hash_input_size, leaf_idx] = merkle_proof.get_leaf();
      HashConfig hash_config;

      // offset in bytes to the selected leaf at the full tree
      uint64_t leaf_element_start = leaf_idx * m_backend->get_leaf_element_size();

      // Hash the leaves into hash_results
      uint hash_output_size = layer_hashes[0].output_size();
      std::vector<std::byte> hash_results(hash_output_size);
      layer_hashes[0].hash(leaf, hash_input_size, hash_config, hash_results.data());

      auto [path, path_size] = merkle_proof.get_path();
      for (int layer_idx = 1; layer_idx < nof_layers; layer_idx++) {
        // update the pointer to the selected leaf by dividing it by the shrinking factor of the hash
        leaf_element_start = (leaf_element_start / hash_input_size) * hash_output_size;

        // Calculate the previous hash result offset inside the current hash input
        hash_input_size = layer_hashes[layer_idx].input_default_chunk_size();
        hash_output_size = layer_hashes[layer_idx].output_size();
        const int element_offset_in_path = leaf_element_start % hash_input_size;

        if (merkle_proof.is_pruned()) {
          // build an input vector to the next hash layer based on the siblings from the path
          const uint siblings_size = hash_input_size - hash_results.size();
          std::vector<std::byte> hash_inputs(path, path + siblings_size);
          path += siblings_size;

          // push the hash results from previous layer to the right offset
          hash_inputs.insert(hash_inputs.begin() + element_offset_in_path, hash_results.begin(), hash_results.end());

          // run current layer hash
          hash_results.resize(hash_output_size);
          layer_hashes[layer_idx].hash(hash_inputs.data(), hash_input_size, hash_config, hash_results.data());
        } else { // not pruned
          // check that the hash result from previous layer matches the element in the path
          if (std::memcmp(hash_results.data(), path + element_offset_in_path, hash_results.size())) {
            // Comparison failed. No need to continue check
            valid = false;
            return eIcicleError::SUCCESS;
          }
          // run current layer hash
          hash_results.resize(hash_output_size);
          layer_hashes[layer_idx].hash(path, hash_input_size, hash_config, hash_results.data());
          path += hash_input_size;
        }
      }

      // Compare the last hash result with the root
      auto [root, root_size] = merkle_proof.get_root();
      valid = !std::memcmp(hash_results.data(), root, root_size);
      return eIcicleError::SUCCESS;
    }

  private:
    std::shared_ptr<MerkleTreeBackend>
      m_backend; ///< Shared pointer to the backend responsible for Merkle tree operations.
  };

} // namespace icicle