
#include "icicle/backend/hash/blake2s_backend.h"

namespace icicle {
  const int BLAKE2S_RATE = 136;
  const int BLAKE2S_DIGEST = 4;
  const int BLAKE2S_PADDING_CONST = 1;

  class Blake2sBackend : public HashBackend
  {
  public:
    Blake2sBackend(uint64_t input_chunk_size)
        : HashBackend("Blake2s-CPU", BLAKE2S_DIGEST, BLAKE2S_RATE)
    {
    }

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      ICICLE_LOG_DEBUG << "Blake2s CPU hash() called, batch=" << config.batch
                       << ", single_output_size=" << output_size();
      // TODO implement real logic
      for (int i = 0; i < output_size() * config.batch; ++i) {
        output[i] = std::byte(i % 256);
      }
      return eIcicleError::SUCCESS;
    }
  };

  /************************ Blake2s registration ************************/
  eIcicleError
  create_blake2s_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Blake2sBackend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_BLAKE2S_FACTORY_BACKEND("CPU", create_blake2s_hash_backend);

} // namespace icicle