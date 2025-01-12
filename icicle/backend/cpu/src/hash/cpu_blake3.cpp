#include "blake3.h"
#include "icicle/backend/hash/blake3_backend.h"
#include "icicle/utils/modifiers.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace icicle {

  class Blake3BackendCPU : public HashBackend
  {
  public:
    Blake3BackendCPU(uint64_t input_chunk_size) : HashBackend("Blake3-CPU", BLAKE3_OUTBYTES, input_chunk_size) {}

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      const auto digest_size_in_bytes = output_size();
      const auto single_input_size = get_single_chunk_size(size);

      // Initialize the hasher
      blake3_hasher hasher;

      for (unsigned batch_idx = 0; batch_idx < config.batch; ++batch_idx) {
        const std::byte* batch_input = input + batch_idx * single_input_size;
        uint64_t batch_size = single_input_size;

        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, reinterpret_cast<const uint8_t*>(batch_input), batch_size);

        uint8_t* batch_output = reinterpret_cast<uint8_t*>(output + batch_idx * digest_size_in_bytes);
        blake3_hasher_finalize(&hasher, batch_output, digest_size_in_bytes);
      }

      return eIcicleError::SUCCESS;
    }

  private:
    static constexpr unsigned int BLAKE3_OUTBYTES = 32; // BLAKE3 default output size in bytes
  };

  /************************ Blake3 registration ************************/
  eIcicleError
  create_blake3_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Blake3BackendCPU>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_BLAKE3_FACTORY_BACKEND("CPU", create_blake3_hash_backend);

} // namespace icicle