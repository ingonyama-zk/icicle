#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/utils/utils.h"

namespace icicle {

  template <typename S>
  struct PoseidonConstantsCPU : PoseidonConstants<S> {
    // TODO add field here
    S* m_dummy_poseidon_constant = nullptr;
    int dummy_int = 79;
  };

  static eIcicleError cpu_poseidon_init_constants(
    const Device& device,
    unsigned arity,
    unsigned alpha,
    unsigned full_rounds_half,
    unsigned partial_rounds,
    const scalar_t* rounds_constants,
    const scalar_t* mds_matrix,
    const scalar_t* non_sparse_matrix,
    const scalar_t* sparse_matrices,
    const scalar_t* domain_tag,
    std::shared_ptr<PoseidonConstants<scalar_t>>& constants /*out*/)
  {
    ICICLE_LOG_INFO << "in cpu_poseidon_init_constants()";
    constants = std::make_shared<PoseidonConstantsCPU<scalar_t>>();
    // TODO implement
    return eIcicleError::SUCCESS;
  }

  REGISTER_POSEIDON_INIT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_constants);

  static eIcicleError cpu_poseidon_init_default_constants(
    const Device& device, std::shared_ptr<PoseidonConstants<scalar_t>>& constants /*out*/)
  {
    ICICLE_LOG_INFO << "in cpu_poseidon_init_default_constants()";
    constants = std::make_shared<PoseidonConstantsCPU<scalar_t>>();
    // TODO implement
    return eIcicleError::SUCCESS;
  }

  REGISTER_POSEIDON_INIT_DEFAULT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_default_constants);

  template <typename S>
  class PoseidonBackendCPU : public HashBackend
  {
  public:
    PoseidonBackendCPU(std::shared_ptr<PoseidonConstants<S>> constants)
        : HashBackend(sizeof(S), 0 /*TODO get from constants arity of whatever*/)
    {
      auto downcast_constants = std::dynamic_pointer_cast<PoseidonConstantsCPU<S>>(constants);
      ICICLE_ASSERT(downcast_constants != nullptr)
        << "CPU poseidon expecting PoseidonConstantsCPU<S> constants, got invalid constant type";
      m_constants = downcast_constants;
      ICICLE_LOG_DEBUG << "constant.dummy_int=" << m_constants->dummy_int;
    }

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      ICICLE_LOG_DEBUG << "Poseidon CPU hash() " << size << " bytes, for type " << demangle<S>()
                       << ", batch=" << config.batch;
      // TODO implement
      return eIcicleError::SUCCESS;
    }

  private:
    std::shared_ptr<PoseidonConstantsCPU<S>> m_constants = nullptr;
  };

  static eIcicleError create_cpu_poseidon_hash_backend(
    const Device& device,
    std::shared_ptr<PoseidonConstants<scalar_t>> constants,
    std::shared_ptr<HashBackend>& backend /*OUT*/)
  {
    backend = std::make_shared<PoseidonBackendCPU<scalar_t>>(constants);
    return eIcicleError::SUCCESS;
  }

  REGISTER_CREATE_POSEIDON_BACKEND("CPU", create_cpu_poseidon_hash_backend);

} // namespace icicle