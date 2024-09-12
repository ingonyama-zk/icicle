#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/errors.h"

namespace icicle {

  class CPUMerkleTreeBackend : public MerkleTreeBackend
  {
  public:
    CPUMerkleTreeBackend(
      const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0)
        : MerkleTreeBackend(layer_hashes, leaf_element_size, output_store_min_layer)
    {
    }

    eIcicleError build(const std::byte* leaves, uint64_t size, const MerkleTreeConfig& config) override
    {
      ICICLE_LOG_INFO << "in CPU CPUMerkleTreeBackend::build()";
      // TODO implement
      return eIcicleError::SUCCESS;
    }

    eIcicleError get_merkle_root(std::byte* root /*output*/) const override
    {
      ICICLE_LOG_INFO << "in CPU CPUMerkleTreeBackend::get_root()";
      return eIcicleError::SUCCESS;
    }

    eIcicleError get_merkle_path(
      const std::byte* leaves,
      uint64_t element_idx,
      const MerkleTreeConfig& config,
      MerklePath& merkle_path /*output*/) const override
    {
      ICICLE_LOG_INFO << "in CPU CPUMerkleTreeBackend::get_path()";
      // TODO implement
      return eIcicleError::SUCCESS;
    }
  };

  eIcicleError create_merkle_tree_backend(
    const Device& device,
    const std::vector<Hash>& layer_hashes,
    uint64_t leaf_element_size,
    uint64_t output_store_min_layer,
    std::shared_ptr<MerkleTreeBackend>& backend /*OUT*/)
  {
    backend = std::make_shared<CPUMerkleTreeBackend>(layer_hashes, leaf_element_size, output_store_min_layer);
    return eIcicleError::SUCCESS;
  }

  REGISTER_MERKLE_TREE_FACTORY_BACKEND("CPU", create_merkle_tree_backend);

} // namespace icicle