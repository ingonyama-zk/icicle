#include "icicle/backend/hash/skyscraper_backend.h"
#include "icicle/fields/field_config.h"
#include <memory>

namespace icicle {

class SkyscraperBackendCPU : public HashBackend {
public:
  SkyscraperBackendCPU(unsigned n, unsigned beta, unsigned s)
    : HashBackend("Skyscraper", /*output_size=*/32, /*default_input_chunk_size=*/64), n_(n), beta_(beta), s_(s) {}

  eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override {
    // TODO: Implement Skyscraper hash logic here
    return eIcicleError::SUCCESS;
  }

private:
  unsigned n_;
  unsigned beta_;
  unsigned s_;
};

static eIcicleError create_skyscraper_hash_backend_cpu(
  const Device& /*device*/,
  unsigned n,
  unsigned beta,
  unsigned s,
  std::shared_ptr<HashBackend>& backend)
{
  backend = std::make_shared<SkyscraperBackendCPU>(n, beta, s);
  return eIcicleError::SUCCESS;
}

REGISTER_SKYSCRAPER_FACTORY_BACKEND("CPU", create_skyscraper_hash_backend_cpu);

} // namespace icicle 