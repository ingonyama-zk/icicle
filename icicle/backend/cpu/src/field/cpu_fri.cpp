#include "icicle/backend/fri_backend.h"
#include "cpu_fri_backend.h"

using namespace field_config;

namespace icicle {

  template <typename F>
  eIcicleError cpu_create_fri_backend(const Device& device, const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree> merkle_trees, std::shared_ptr<FriBackend<F>>& backend /*OUT*/)
  {
    backend = std::make_shared<CpuFriBackend<F>>(folding_factor, stopping_degree, merkle_trees);
    return eIcicleError::SUCCESS;
  }
  REGISTER_FRI_FACTORY_BACKEND("CPU", cpu_create_fri_backend<scalar_t>);

} // namespace icicle