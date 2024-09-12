#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "icicle/hash/hash.h"
#include "icicle/merkle/merkle_tree_config.h"
#include "icicle/merkle/merkle_path.h"

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
    }

    virtual ~MerkleTreeBackend() = default;

    /**
     * @brief Build the Merkle tree from the provided leaves.
     * @param leaves Pointer to the leaves of the tree (input data).
     * @param size The size of the leaves.
     * @param config Configuration for the Merkle tree operation.
     * @return Error code of type eIcicleError indicating success or failure.
     */
    virtual eIcicleError build(const std::byte* leaves, uint64_t size, const MerkleTreeConfig& config) = 0;

    /**
     * @brief Retrieve the root of the Merkle tree.
     * @param root Pointer to where the Merkle root will be written.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_merkle_root(std::byte* root /*output*/) const = 0;

    /**
     * @brief Retrieve the Merkle path for a specific element.
     * @param leaves Pointer to the leaves of the tree.
     * @param element_idx Index of the element for which the Merkle path is required.
     * @param config Configuration for the Merkle tree operation.
     * @param merkle_path Reference to the MerklePath object where the path will be stored.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_merkle_path(
      const std::byte* leaves,
      uint64_t element_idx,
      const MerkleTreeConfig& config,
      MerklePath& merkle_path /*output*/) const = 0;

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

  private:
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