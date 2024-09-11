#pragma once

#include <functional>

#include "icicle/common.h"
#include "icicle/hash/hash.h"
#include "icicle/merkle/merkle_tree_config.h"

namespace icicle {

  /**
   * @brief Abstract class representing a backend for Merkle tree operations.
   *
   * This backend will handle the actual logic for building, verifying, and retrieving
   * elements in the Merkle tree. Specific device implementations (e.g., CPU, GPU) will derive from this class.
   */
  class MerkleTreeBackend
  {
  public:
    /**
     * @brief Constructor for the MerkleTreeBackend class.
     *
     * @param layer_hashes Vector of Hash objects representing the hashes for each layer.
     * @param leaf_element_size_in_limbs The size of each leaf element in limbs.
     * @param output_store_min_layer Minimum layer index to store in the output.
     */
    MerkleTreeBackend(
      const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size_in_limbs, uint64_t output_store_min_layer = 0)
        : m_layer_hashes(layer_hashes), m_leaf_element_size_in_limbs(leaf_element_size_in_limbs),
          m_output_store_min_layer(output_store_min_layer)
    {
    }

    virtual ~MerkleTreeBackend() = default;

    // Pure virtual methods to be implemented by derived classes
    virtual eIcicleError
    build(const limb_t* leaves, const MerkleTreeConfig& config, const limb_t* secondary_leaves = nullptr) = 0;
    virtual eIcicleError get_root(const limb_t*& root) const = 0;
    virtual eIcicleError
    get_path(const limb_t* leaves, uint64_t element_idx, limb_t* path, const MerkleTreeConfig& config) const = 0;
    virtual eIcicleError
    verify(const limb_t* path, uint64_t element_idx, bool& verification_valid, const MerkleTreeConfig& config) = 0;

    /**
     * @brief Get the hashes used in each layer of the Merkle tree.
     *
     * @return The vector of Hash objects for each layer.
     */
    const std::vector<Hash>& get_layer_hashes() const { return m_layer_hashes; }

    /**
     * @brief Get the size of the leaf elements in limbs.
     *
     * @return The size of each leaf element in limbs.
     */
    uint64_t get_leaf_element_size_in_limbs() const { return m_leaf_element_size_in_limbs; }

    /**
     * @brief Get the minimum layer index to store in the output.
     *
     * @return The minimum layer index to store in the output.
     */
    uint64_t get_output_store_min_layer() const { return m_output_store_min_layer; }

  private:
    std::vector<Hash> m_layer_hashes;      ///< Vector of hashes for each layer.
    uint64_t m_leaf_element_size_in_limbs; ///< Size of each leaf element in limbs.
    uint64_t m_output_store_min_layer;     ///< Minimum layer index to store in the output.
  };

  /*************************** Backend registration ***************************/
  using MerkleTreeFactoryImpl = std::function<eIcicleError(
    const Device& device,
    const std::vector<Hash>& layer_hashes,
    uint64_t leaf_element_size_in_limbs,
    uint64_t output_store_min_layer,
    std::shared_ptr<MerkleTreeBackend>& backend /*OUT*/)>;

  void register_merkle_tree_factory(const std::string& deviceType, MerkleTreeFactoryImpl impl);

#define REGISTER_MERKLE_TREE_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_merkle_tree) = []() -> bool {                                                              \
      register_merkle_tree_factory(DEVICE_TYPE, FUNC);                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle