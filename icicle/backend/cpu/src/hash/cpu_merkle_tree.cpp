#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"

namespace icicle {

  class CPUMerkleTreeBackend : public MerkleTreeBackend
  {
  public:
    CPUMerkleTreeBackend(
      const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0)
        : MerkleTreeBackend(layer_hashes, leaf_element_size, output_store_min_layer)
    {
      ICICLE_LOG_INFO << "in CPUMerkleTreeBackend, have " << layer_hashes.size() << " layer hashes";
      for (const auto& layer_hash : layer_hashes) {
        ICICLE_LOG_INFO << "layer hash (" << &layer_hash << "), name: " << layer_hash.name();
      }
    }

    eIcicleError build(const std::byte* leaves, uint64_t size, const MerkleTreeConfig& config) override
    {
      ICICLE_LOG_INFO << "CPU CPUMerkleTreeBackend::build() called with " << size << " bytes of leaves";
      return eIcicleError::SUCCESS; // TODO: Implement tree-building logic
    }

    std::pair<std::byte*, size_t> get_merkle_root() const override
    {
      ICICLE_LOG_INFO << "CPU CPUMerkleTreeBackend::get_merkle_root() called";
      return {nullptr, 0}; // TODO: Implement root retrieval logic
    }

    eIcicleError get_merkle_proof(
      const std::byte* leaves,
      uint64_t element_idx,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof) const override
    {
      ICICLE_LOG_INFO << "CPU CPUMerkleTreeBackend::get_merkle_proof() called for element index " << element_idx;
      //     void allocate(
      // bool pruned_path,
      // uint64_t leaf_idx,
      // const std::byte* leaf,
      // std::size_t leaf_size,
      // const std::byte* root,
      // std::size_t root_size)

      uint64_t root = 123;
      merkle_proof.allocate(
        false /*pruned*/, element_idx, leaves + element_idx * get_leaf_element_size(), get_leaf_element_size(),
        (const std::byte*)&root, sizeof(root));
      return eIcicleError::SUCCESS; // TODO: Implement proof generation logic
    }
  };

  eIcicleError create_merkle_tree_backend(
    const Device& device,
    const std::vector<Hash>& layer_hashes,
    uint64_t leaf_element_size,
    uint64_t output_store_min_layer,
    std::shared_ptr<MerkleTreeBackend>& backend)
  {
    ICICLE_LOG_INFO << "Creating CPU MerkleTreeBackend";
    backend = std::make_shared<CPUMerkleTreeBackend>(layer_hashes, leaf_element_size, output_store_min_layer);
    return eIcicleError::SUCCESS;
  }

  REGISTER_MERKLE_TREE_FACTORY_BACKEND("CPU", create_merkle_tree_backend);

} // namespace icicle