/*BLAKE3 Hash function based on the original design by the BLAKE3 team https://github.com/BLAKE3-team/BLAKE3 */

#include "blake3.h"
#include "icicle/backend/hash/blake3_backend.h"
#include "icicle/utils/modifiers.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include "icicle/config_extension.h"

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

      size_t num_chunks = 1;
      if (config.ext && config.ext->has(CpuBackendConfig::CPU_NOF_THREADS)) {
        num_chunks = config.ext && (config.ext->get<int>(CpuBackendConfig::CPU_NOF_THREADS) != 0)
                       ? config.ext->get<int>(CpuBackendConfig::CPU_NOF_THREADS)
                       :                                    // number of threads provided by config
                       std::thread::hardware_concurrency(); // check machine properties (if provided with 0)
      }
      if (num_chunks <= 0) {
        ICICLE_LOG_WARNING << "Unable to detect number of hardware supported threads - fixing it to 1\n";
        num_chunks = 1;
      }

      size_t chunk_size = (config.batch + num_chunks - 1) / num_chunks;

      if (num_chunks == 1) { // single thread without using taskflow
        for (unsigned batch_idx = 0; batch_idx < config.batch; ++batch_idx) {
          const std::byte* batch_input = input + batch_idx * single_input_size;
          uint64_t batch_size = single_input_size;
          blake3_hasher hasher;
          blake3_hasher_init(&hasher);
          blake3_hasher_update(&hasher, reinterpret_cast<const uint8_t*>(batch_input), batch_size);
          uint8_t* batch_output = reinterpret_cast<uint8_t*>(output + batch_idx * digest_size_in_bytes);
          blake3_hasher_finalize(&hasher, batch_output, digest_size_in_bytes);
        }
      } else {
        tf::Taskflow taskflow;
        tf::Executor executor;
        for (size_t i = 0; i < num_chunks; ++i) {
          size_t start_index = i * chunk_size;
          size_t end_index = std::min(start_index + chunk_size, static_cast<size_t>(config.batch));
          taskflow.emplace([&, start_index, end_index, output, digest_size_in_bytes, single_input_size, input]() {
            for (unsigned batch_idx = start_index; batch_idx < end_index; ++batch_idx) {
              blake3_hasher hasher;
              blake3_hasher_init(&hasher);
              blake3_hasher_update(&hasher, input + batch_idx * single_input_size, single_input_size);
              blake3_hasher_finalize(
                &hasher, reinterpret_cast<uint8_t*>(output + batch_idx * digest_size_in_bytes), digest_size_in_bytes);
            }
          });
        }
        executor.run(taskflow).wait();
      }
      return eIcicleError::SUCCESS;
    }

  private:
    static constexpr unsigned int BLAKE3_OUTBYTES = 32; // BLAKE3 default output size in bytes
  };

  /************************ Blake3 registration ************************/
  eIcicleError
  create_blake3_hash_backend_cpu(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Blake3BackendCPU>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_BLAKE3_FACTORY_BACKEND("CPU", create_blake3_hash_backend_cpu);

} // namespace icicle