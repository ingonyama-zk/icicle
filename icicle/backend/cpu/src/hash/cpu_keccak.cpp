
#include "icicle/backend/hash/keccak_backend.h"

namespace icicle {

  class KeccakBackend : public HashBackend
  {
  public:
    KeccakBackend(int total_input_limbs, int total_output_limbs, unsigned int rate, unsigned int padding_const)
        : HashBackend(total_input_limbs, total_output_limbs)
    {
    }

    eIcicleError hash_single(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      const HashConfig& config) const override
    {
      ICICLE_LOG_INFO << "Keccak CPU hash_single() called";
      // TODO implement
      return eIcicleError::SUCCESS;
    }

    eIcicleError hash_many(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config) const override
    {
      ICICLE_LOG_INFO << "Keccak CPU hash_many() called";
      // TODO implement
      return eIcicleError::SUCCESS;
    }
  };

  const int KECCAK_256_RATE = 136;
  const int KECCAK_256_DIGEST = 4;
  const int KECCAK_512_RATE = 72;
  const int KECCAK_512_DIGEST = 8;
  const int KECCAK_STATE_SIZE = 25;
  const int KECCAK_PADDING_CONST = 1;
  const int SHA3_PADDING_CONST = 6;

  class Keccak256Backend : public KeccakBackend
  {
  public:
    Keccak256Backend(int total_input_limbs)
        : KeccakBackend(
            total_input_limbs,
            KECCAK_256_DIGEST * sizeof(uint64_t) / sizeof(limb_t),
            KECCAK_256_RATE,
            KECCAK_PADDING_CONST)
    {
    }
  };

  class Keccak512Backend : public KeccakBackend
  {
  public:
    Keccak512Backend(int total_input_limbs)
        : KeccakBackend(
            total_input_limbs,
            KECCAK_512_DIGEST * sizeof(uint64_t) / sizeof(limb_t),
            KECCAK_512_RATE,
            KECCAK_PADDING_CONST)
    {
    }
  };

  class Sha3_256Backend : public KeccakBackend
  {
  public:
    Sha3_256Backend(int total_input_limbs)
        : KeccakBackend(
            total_input_limbs,
            KECCAK_256_DIGEST * sizeof(uint64_t) / sizeof(limb_t),
            KECCAK_256_RATE,
            SHA3_PADDING_CONST)
    {
    }
  };

  class Sha3_512Backend : public KeccakBackend
  {
  public:
    Sha3_512Backend(int total_input_limbs)
        : KeccakBackend(
            total_input_limbs,
            KECCAK_512_DIGEST * sizeof(uint64_t) / sizeof(limb_t),
            KECCAK_512_RATE,
            SHA3_PADDING_CONST)
    {
    }
  };

  /************************ Keccak 256 registration ************************/
  eIcicleError create_keccak_256_hash_backend(
    const Device& device, uint64_t total_input_limbs, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Keccak256Backend>(total_input_limbs);
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_256_FACTORY_BACKEND("CPU", create_keccak_256_hash_backend);

  /************************ Keccak 512 registration ************************/
  eIcicleError create_keccak_512_hash_backend(
    const Device& device, uint64_t total_input_limbs, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Keccak512Backend>(total_input_limbs);
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_512_FACTORY_BACKEND("CPU", create_keccak_512_hash_backend);

  /************************ SHA3 256 registration ************************/
  eIcicleError
  create_sha3_256_hash_backend(const Device& device, uint64_t total_input_limbs, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Sha3_256Backend>(total_input_limbs);
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_256_FACTORY_BACKEND("CPU", create_sha3_256_hash_backend);

  /************************ SHA3 512 registration ************************/
  eIcicleError
  create_sha3_512_hash_backend(const Device& device, uint64_t total_input_limbs, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Sha3_512Backend>(total_input_limbs);
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_512_FACTORY_BACKEND("CPU", create_sha3_512_hash_backend);

} // namespace icicle