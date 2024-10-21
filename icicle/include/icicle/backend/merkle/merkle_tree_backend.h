#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "icicle/hash/hash.h"
#include "icicle/merkle/merkle_tree_config.h"
#include "icicle/merkle/merkle_proof.h"

namespace icicle {

  /**
   * @brief Abstract base class for Merkle tree backend implementations.
   *
   * This backend handles the core logic for Merkle tree operations such as building the tree,
   * retrieving the root, computing Merkle paths, and verifying elements. Derived classes
   * will provide specific implementations for various devices (e.g., CPU, GPU).
   */
  class MerkleTreeBackend
  {
  public:
    /**
     * @brief Constructor for the MerkleTreeBackend class.
     *
     * @param layer_hashes Vector of Hash objects representing the hash function for each layer.
     * @param leaf_element_size Size of each leaf element in bytes.
     * @param output_store_min_layer Minimum layer index to store in the output (default is 0).
     */
    MerkleTreeBackend(
      const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0)
        : m_layer_hashes(layer_hashes), m_leaf_element_size(leaf_element_size),
          m_output_store_min_layer(output_store_min_layer)
    {
      ICICLE_ASSERT(output_store_min_layer < layer_hashes.size())
        << "output_store_min_layer must be smaller than nof_layers. At least the root should be saved on tree. "
           "(nof_layers="
        << layer_hashes.size() << ", output_store_min_layer=" << output_store_min_layer << ")\n";

      ICICLE_ASSERT(layer_hashes[0].input_default_chunk_size() % leaf_element_size == 0)
        << "A whole number of leaves must be fitted into the hashes of the first layer.\n";
    }

    virtual ~MerkleTreeBackend() = default;

    /**
     * @brief Build the Merkle tree from the provided leaves.
     * @param leaves Pointer to the leaves of the tree (input data).
     * @param leaves_size The size of the leaves.
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    virtual eIcicleError build(const std::byte* leaves, uint64_t leaves_size, const MerkleTreeConfig& config) = 0;

    /**
     * @brief Returns a pair containing the pointer to the root (ON HOST) data and its size.
     * @return A pair of (root data pointer, root size).
     */
    virtual std::pair<const std::byte*, size_t> get_merkle_root() const = 0;

    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param size The size of the leaves.
     * @param leaf_idx Index of the leaf element for which the Merkle path is required.
     * @param is_pruned If set, the path will not include hash results that can be extracted from siblings
     * @param config Configuration for the Merkle tree operation.
     * @param merkle_proof Reference to the MerkleProof object where the path will be stored.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_merkle_proof(
      const std::byte* leaves,
      uint64_t size,
      uint64_t leaf_idx,
      bool is_pruned,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof /*output*/) const = 0;

    /**
     * @brief Get the hash functions used for each layer of the Merkle tree.
     *
     * @return A vector of Hash objects representing the hash function for each layer.
     */
    const std::vector<Hash>& get_layer_hashes() const { return m_layer_hashes; }

    /**
     * @brief Get the size of each leaf element in bytes.
     *
     * @return Size of each leaf element in bytes.
     */
    uint64_t get_leaf_element_size() const { return m_leaf_element_size; }

    /**
     * @brief Get the minimum layer index to store in the output.
     *
     * @return Minimum layer index to store in the output.
     */
    uint64_t get_output_store_min_layer() const { return m_output_store_min_layer; }

  protected:
    std::vector<Hash> m_layer_hashes;  ///< Vector of hash functions for each layer.
    uint64_t m_leaf_element_size;      ///< Size of each leaf element in bytes.
    uint64_t m_output_store_min_layer; ///< Minimum layer index to store in the output.
  };

  /*************************** Backend Factory Registration ***************************/

  using MerkleTreeFactoryImpl = std::function<eIcicleError(
    const Device& device,
    const std::vector<Hash>& layer_hashes,
    uint64_t leaf_element_size,
    uint64_t output_store_min_layer,
    std::shared_ptr<MerkleTreeBackend>& backend /*OUT*/)>;

  /**
   * @brief Register a MerkleTree backend factory for a specific device type.
   *
   * @param deviceType String identifier for the device type.
   * @param impl Factory function that creates the MerkleTreeBackend.
   */
  void register_merkle_tree_factory(const std::string& deviceType, MerkleTreeFactoryImpl impl);

  /**
   * @brief Macro to register a MerkleTree backend factory.
   *
   * This macro registers a factory function for a specific backend by calling
   * `register_merkle_tree_factory` at runtime.
   */
#define REGISTER_MERKLE_TREE_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_merkle_tree) = []() -> bool {                                                              \
      register_merkle_tree_factory(DEVICE_TYPE, FUNC);                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle