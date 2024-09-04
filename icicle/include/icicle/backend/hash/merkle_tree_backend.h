#pragma once

#include <functional>

#include "icicle/common.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/merkle_tree_config.h"

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
     * @param layer_hashes Vector of hashes for each layer.
     * @param output_store_min_layer Minimum layer index to store in the output.
     */
    MerkleTreeBackend(
      const std::vector<Hash>& layer_hashes,
      const unsigned int leaf_element_size_in_limbs,
      const unsigned int output_store_min_layer = 0);

    virtual ~MerkleTreeBackend() = default;

    virtual eIcicleError
    build(const limb_t* leaves, const MerkleTreeConfig& config, const limb_t* secondary_leaves = nullptr) = 0;
    virtual eIcicleError get_root(const limb_t*& root) const = 0;
    virtual eIcicleError
    get_path(const limb_t* leaves, uint64_t element_idx, limb_t* path, const MerkleTreeConfig& config) const = 0;
    virtual eIcicleError
    verify(const limb_t* path, unsigned int element_idx, bool& verification_valid, const MerkleTreeConfig& config) = 0;
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