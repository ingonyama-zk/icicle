#include "icicle/backend/hash/skyscraper_backend.h"
#include "icicle/fields/field_config.h"
#include "icicle/fields/field.h"
#include <memory>
#include <vector>

#if FIELD_ID == BN254
  #include "icicle/hash/skyscraper_constants/constants/bn254_skyscraper.h"
  using namespace skyscraper_constants_bn254;
#elif FIELD_ID == BLS12_381
  #include "icicle/hash/skyscraper_constants/constants/bls12_381_skyscraper.h"
  using namespace skyscraper_constants_bls12_381;
#endif

namespace icicle {

  static eIcicleError init_default_constants(unsigned n) {
    unsigned char* constants = nullptr;

    switch (n) {
      case 1:
        constants = round_constants_1;
        break;
      case 2:
        constants = round_constants_2;
        break;
      case 3:
        constants = round_constants_3;
        break;
      default:
        ICICLE_LOG_ERROR << "Skyscraper: unsupported n=" << n;
        break;
    }
    
    scalar_t* h_constants = reinterpret_cast<scalar_t*>(constants);

    return eIcicleError::SUCCESS;
  }

class SkyscraperBackendCPU : public HashBackend {
public:
  SkyscraperBackendCPU(unsigned n, unsigned beta, unsigned s)
    : HashBackend("Skyscraper", sizeof(scalar_t), /*input_chunk_size=*/n*2), n_(n), beta_(beta), s_(s) {
      init_default_constants(n);
  }

  eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override {
    // TODO: Convert input to scalar_t array
    // TODO: Call hash_single for each batch
    return eIcicleError::SUCCESS;
  }

private:
  eIcicleError hash_single(const scalar_t* input, unsigned input_size, scalar_t* output) const {
    // 1. convert input to field elements
    // 2. call compress
    return eIcicleError::SUCCESS;
  }

  // Skyscraper permutation (Feistel)
  void skyscraper_permutation(scalar_t* state) const {
    // TODO: feistel rounds as in Sage
  }

  // Bar function
  scalar_t bar(const scalar_t& x) const {
    // TODO: bar as in Sage
    return x;
  }

  // S-box function
  scalar_t sbox(const scalar_t& x) const {
    // TODO: s-box as in Sage
    return x;
  }

  // Compression function (2n -> n)
  void compress(const scalar_t* input, scalar_t* output) const {
    // TODO: compress as in Sage
  }

  // Round constants
  std::vector<scalar_t> round_constants_;

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