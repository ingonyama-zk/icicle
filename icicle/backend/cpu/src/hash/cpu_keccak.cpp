
#include "icicle/backend/hash/keccak_backend.h"

namespace icicle {

  class KeccakBackend : public HashBackend
  {
  public:
    KeccakBackend(uint64_t input_chunk_size, uint64_t output_size, uint64_t rate, int padding_const)
        : HashBackend(output_size, input_chunk_size)
    {
    }

    eIcicleError
    hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output_limbs) const override
    {
      ICICLE_LOG_INFO << "Keccak CPU hash() called";
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
    Keccak256Backend(int input_chunk_size)
        : KeccakBackend(
            input_chunk_size,
            KECCAK_256_DIGEST * sizeof(uint64_t) / sizeof(std::byte),
            KECCAK_256_RATE,
            KECCAK_PADDING_CONST)
    {
    }
  };

  class Keccak512Backend : public KeccakBackend
  {
  public:
    Keccak512Backend(int input_chunk_size)
        : KeccakBackend(
            input_chunk_size,
            KECCAK_512_DIGEST * sizeof(uint64_t) / sizeof(std::byte),
            KECCAK_512_RATE,
            KECCAK_PADDING_CONST)
    {
    }
  };

  class Sha3_256Backend : public KeccakBackend
  {
  public:
    Sha3_256Backend(int input_chunk_size)
        : KeccakBackend(
            input_chunk_size,
            KECCAK_256_DIGEST * sizeof(uint64_t) / sizeof(std::byte),
            KECCAK_256_RATE,
            SHA3_PADDING_CONST)
    {
    }
  };

  class Sha3_512Backend : public KeccakBackend
  {
  public:
    Sha3_512Backend(int input_chunk_size)
        : KeccakBackend(
            input_chunk_size,
            KECCAK_512_DIGEST * sizeof(uint64_t) / sizeof(std::byte),
            KECCAK_512_RATE,
            SHA3_PADDING_CONST)
    {
    }
  };

  /************************ Keccak 256 registration ************************/
  eIcicleError
  create_keccak_256_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Keccak256Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_256_FACTORY_BACKEND("CPU", create_keccak_256_hash_backend);

  /************************ Keccak 512 registration ************************/
  eIcicleError
  create_keccak_512_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Keccak512Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_512_FACTORY_BACKEND("CPU", create_keccak_512_hash_backend);

  /************************ SHA3 256 registration ************************/
  eIcicleError
  create_sha3_256_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Sha3_256Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_256_FACTORY_BACKEND("CPU", create_sha3_256_hash_backend);

  /************************ SHA3 512 registration ************************/
  eIcicleError
  create_sha3_512_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Sha3_512Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_512_FACTORY_BACKEND("CPU", create_sha3_512_hash_backend);

} // namespace icicle