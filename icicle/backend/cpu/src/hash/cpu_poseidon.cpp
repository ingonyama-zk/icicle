#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/utils/utils.h"

namespace icicle {

  static eIcicleError
  cpu_poseidon_init_constants(const Device& device, const PoseidonConstantsInitOptions<scalar_t>* options)
  {
    ICICLE_LOG_DEBUG << "in cpu_poseidon_init_constants()";
    // TODO implement
    return eIcicleError::SUCCESS;
  }

  REGISTER_POSEIDON_INIT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_constants);

  static eIcicleError cpu_poseidon_init_default_constants(const Device& device, const scalar_t& phantom)
  {
    ICICLE_LOG_DEBUG << "in cpu_poseidon_init_default_constants()";
    // TODO implement
    return eIcicleError::SUCCESS;
  }

  REGISTER_POSEIDON_INIT_DEFAULT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_default_constants);

  template <typename S>
  class PoseidonBackendCPU : public HashBackend
  {
  public:
    PoseidonBackendCPU(unsigned arity) : HashBackend("Poseidon-CPU", sizeof(S), arity * sizeof(S)) {}

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      ICICLE_LOG_DEBUG << "Poseidon CPU hash() " << size << " bytes, for type " << demangle<S>()
                       << ", batch=" << config.batch;
      // TODO implement
      return eIcicleError::SUCCESS;
    }
  };

  static eIcicleError create_cpu_poseidon_hash_backend(
    const Device& device, unsigned arity, std::shared_ptr<HashBackend>& backend /*OUT*/, const scalar_t& phantom)
  {
    ICICLE_LOG_DEBUG << "in create_cpu_poseidon_hash_backend(arity=" << arity << ")";
    backend = std::make_shared<PoseidonBackendCPU<scalar_t>>(arity);
    return eIcicleError::SUCCESS;
  }

  REGISTER_CREATE_POSEIDON_BACKEND("CPU", create_cpu_poseidon_hash_backend);

} // namespace icicle