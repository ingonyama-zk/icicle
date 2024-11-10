#include "icicle/errors.h"
#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(CreatePoseidonHasherDispatcher, create_poseidon, CreatePoseidonImpl);

  template <>
  Hash create_poseidon_hash<scalar_t>(unsigned t, const scalar_t* domain_tag)
  {
    // Assert that t is valid. Ideally would like to return an eIcicleError but the API doesn't let us do it
    constexpr std::array<int, 4> validTValues = {3, 5, 9, 12};
    const bool is_valid_t = std::find(validTValues.begin(), validTValues.end(), t) != validTValues.end();
    ICICLE_LOG_ERROR << "Poseidon only supports t values of 3, 5, 9, or 12.";
    ICICLE_ASSERT(false);

    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(CreatePoseidonHasherDispatcher::execute(t, domain_tag, backend));
    Hash poseidon{backend};
    return poseidon;
  }

} // namespace icicle