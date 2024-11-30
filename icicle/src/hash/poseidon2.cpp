#include "icicle/errors.h"
#include "icicle/backend/hash/poseidon2_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(CreatePoseidon2HasherDispatcher, create_poseidon2, CreatePoseidon2Impl);

  template <>
  Hash create_poseidon2_hash<scalar_t>(unsigned t, const scalar_t* domain_tag)
  {
    // Assert that t is valid. Ideally would like to return an eIcicleError but the API doesn't let us do it
    constexpr std::array<int, 8> validTValues = {2, 3, 4, 8, 12, 16, 20, 24};
    const bool is_valid_t = std::find(validTValues.begin(), validTValues.end(), t) != validTValues.end();
    if (!is_valid_t) {
      ICICLE_LOG_ERROR << "Poseidon2 only supports t values of 2, 3, 4, 8, 12, 16, 20, 24.";
      ICICLE_ASSERT(false);
    }

    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(CreatePoseidon2HasherDispatcher::execute(t, domain_tag, backend));
    Hash poseidon2{backend};
    return poseidon2;
  }

} // namespace icicle