#include "icicle/pqc/ml_kem.h"

using namespace icicle;
using namespace icicle::pqc::ml_kem;

namespace icicle::pqc::ml_kem {

  extern "C" {
  eIcicleError icicle_ml_kem_keygen512(
    const std::byte* entropy, const MlKemConfig* config, std::byte* public_key, std::byte* secret_key)
  {
    return keygen<Kyber512Params>(entropy, *config, public_key, secret_key);
  }

  eIcicleError icicle_ml_kem_encapsulate512(
    const std::byte* message,
    const std::byte* public_key,
    const MlKemConfig* config,
    std::byte* ciphertext,
    std::byte* shared_secret)
  {
    return encapsulate<Kyber512Params>(message, public_key, *config, ciphertext, shared_secret);
  }

  eIcicleError icicle_ml_kem_decapsulate512(
    const std::byte* secret_key, const std::byte* ciphertext, const MlKemConfig* config, std::byte* shared_secret)
  {
    return decapsulate<Kyber512Params>(secret_key, ciphertext, *config, shared_secret);
  }

  eIcicleError icicle_ml_kem_keygen768(
    const std::byte* entropy, const MlKemConfig* config, std::byte* public_key, std::byte* secret_key)
  {
    return keygen<Kyber768Params>(entropy, *config, public_key, secret_key);
  }

  eIcicleError icicle_ml_kem_encapsulate768(
    const std::byte* message,
    const std::byte* public_key,
    const MlKemConfig* config,
    std::byte* ciphertext,
    std::byte* shared_secret)
  {
    return encapsulate<Kyber768Params>(message, public_key, *config, ciphertext, shared_secret);
  }

  eIcicleError icicle_ml_kem_decapsulate768(
    const std::byte* secret_key, const std::byte* ciphertext, const MlKemConfig* config, std::byte* shared_secret)
  {
    return decapsulate<Kyber768Params>(secret_key, ciphertext, *config, shared_secret);
  }

  eIcicleError icicle_ml_kem_keygen1024(
    const std::byte* entropy, const MlKemConfig* config, std::byte* public_key, std::byte* secret_key)
  {
    return keygen<Kyber1024Params>(entropy, *config, public_key, secret_key);
  }

  eIcicleError icicle_ml_kem_encapsulate1024(
    const std::byte* message,
    const std::byte* public_key,
    const MlKemConfig* config,
    std::byte* ciphertext,
    std::byte* shared_secret)
  {
    return encapsulate<Kyber1024Params>(message, public_key, *config, ciphertext, shared_secret);
  }

  eIcicleError icicle_ml_kem_decapsulate1024(
    const std::byte* secret_key, const std::byte* ciphertext, const MlKemConfig* config, std::byte* shared_secret)
  {
    return decapsulate<Kyber1024Params>(secret_key, ciphertext, *config, shared_secret);
  }
  }
} // namespace icicle::pqc::ml_kem