#include "icicle/backend/fri_backend.h"
#include "cpu_fri_backend.h"

using namespace field_config;
using namespace icicle;

namespace icicle {

  template <typename S, typename F>
  eIcicleError cpu_create_fri_backend(
    const Device& device,
    const size_t folding_factor,
    const size_t stopping_degree,
    std::vector<MerkleTree> merkle_trees,
    std::shared_ptr<FriBackend<S, F>>& backend /*OUT*/)
  {
    backend = std::make_shared<CpuFriBackend<S, F>>(folding_factor, stopping_degree, std::move(merkle_trees));
    return eIcicleError::SUCCESS;
  }

  REGISTER_FRI_FACTORY_BACKEND("CPU", (cpu_create_fri_backend<scalar_t, scalar_t>));
#ifdef EXT_FIELD
  REGISTER_FRI_EXT_FACTORY_BACKEND("CPU", (cpu_create_fri_backend<scalar_t, extension_t>));
#endif // EXT_FIELD

} // namespace icicle