#include "icicle/pqc/ml_kem.h"

#include "icicle/errors.h"
#include "icicle/dispatcher.h"
#include "icicle/backend/pqc/ml_kem_backend.h"

namespace icicle {
  namespace pqc {
    namespace ml_kem {
      ICICLE_DISPATCHER_INST(MlKemKeygen512Dispatcher, ml_kem_keygen512, MlKemKeygenImpl);
      ICICLE_DISPATCHER_INST(MlKemEncapsulate512Dispatcher, ml_kem_encaps512, MlKemEncapsulateImpl);
      ICICLE_DISPATCHER_INST(MlKemDecapsulate512Dispatcher, ml_kem_decaps512, MlKemDecapsulateImpl);

      ICICLE_DISPATCHER_INST(MlKemKeygen768Dispatcher, ml_kem_keygen768, MlKemKeygenImpl);
      ICICLE_DISPATCHER_INST(MlKemEncapsulate768Dispatcher, ml_kem_encaps768, MlKemEncapsulateImpl);
      ICICLE_DISPATCHER_INST(MlKemDecapsulate768Dispatcher, ml_kem_decaps768, MlKemDecapsulateImpl);

      ICICLE_DISPATCHER_INST(MlKemKeygen1024Dispatcher, ml_kem_keygen1024, MlKemKeygenImpl);
      ICICLE_DISPATCHER_INST(MlKemEncapsulate1024Dispatcher, ml_kem_encaps1024, MlKemEncapsulateImpl);
      ICICLE_DISPATCHER_INST(MlKemDecapsulate1024Dispatcher, ml_kem_decaps1024, MlKemDecapsulateImpl);

      template <>
      eIcicleError keygen<Kyber512Params>(
        const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
      {
        return MlKemKeygen512Dispatcher::execute(entropy, config, public_keys, secret_keys);
      }

      template <>
      eIcicleError encapsulate<Kyber512Params>(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulate512Dispatcher::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      template <>
      eIcicleError decapsulate<Kyber512Params>(
        const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
      {
        return MlKemDecapsulate512Dispatcher::execute(secret_keys, ciphertext, config, shared_secrets);
      }

      template <>
      eIcicleError keygen<Kyber768Params>(
        const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
      {
        return MlKemKeygen768Dispatcher::execute(entropy, config, public_keys, secret_keys);
      }

      template <>
      eIcicleError encapsulate<Kyber768Params>(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulate768Dispatcher::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      template <>
      eIcicleError decapsulate<Kyber768Params>(
        const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
      {
        return MlKemDecapsulate768Dispatcher::execute(secret_keys, ciphertext, config, shared_secrets);
      }

      template <>
      eIcicleError keygen<Kyber1024Params>(
        const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
      {
        return MlKemKeygen1024Dispatcher::execute(entropy, config, public_keys, secret_keys);
      }

      template <>
      eIcicleError encapsulate<Kyber1024Params>(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulate1024Dispatcher::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      template <>
      eIcicleError decapsulate<Kyber1024Params>(
        const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
      {
        return MlKemDecapsulate1024Dispatcher::execute(secret_keys, ciphertext, config, shared_secrets);
      }

    } // namespace ml_kem
  } // namespace pqc

} // namespace icicle