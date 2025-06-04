#include "icicle/pqc/ml_kem.h"

#include "icicle/errors.h"
#include "icicle/dispatcher.h"
#include "icicle/backend/pqc/ml_kem_backend.h"

namespace icicle {
  namespace pqc {
    namespace ml_kem {
      ICICLE_DISPATCHER_INST(MlKemKeygenDispatcher, ml_kem_keygen, MlKemKeygenImpl);
      ICICLE_DISPATCHER_INST(MlKemEncapsulateDispatcher, ml_kem_encaps, MlKemEncapsulateImpl);
      ICICLE_DISPATCHER_INST(MlKemDecapsulateDispatcher, ml_kem_decaps, MlKemDecapsulateImpl);

      template <>
      eIcicleError keygen<Kyber512Params>(
        const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
      {
        return MlKemKeygenDispatcher::execute(entropy, config, public_keys, secret_keys);
      }

      template <>
      eIcicleError encapsulate<Kyber512Params>(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulateDispatcher::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      template <>
      eIcicleError decapsulate<Kyber512Params>(
        const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
      {
        return MlKemDecapsulateDispatcher::execute(secret_keys, ciphertext, config, shared_secrets);
      }

      template <>
      eIcicleError keygen<Kyber768Params>(
        const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
      {
        return MlKemKeygenDispatcher::execute(entropy, config, public_keys, secret_keys);
      }

      template <>
      eIcicleError encapsulate<Kyber768Params>(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulateDispatcher::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      template <>
      eIcicleError decapsulate<Kyber768Params>(
        const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
      {
        return MlKemDecapsulateDispatcher::execute(secret_keys, ciphertext, config, shared_secrets);
      }

      template <>
      eIcicleError keygen<Kyber1024Params>(
        const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
      {
        return MlKemKeygenDispatcher::execute(entropy, config, public_keys, secret_keys);
      }

      template <>
      eIcicleError encapsulate<Kyber1024Params>(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulateDispatcher::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      template <>
      eIcicleError decapsulate<Kyber1024Params>(
        const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
      {
        return MlKemDecapsulateDispatcher::execute(secret_keys, ciphertext, config, shared_secrets);
      }

      // TODO: C FFI

    } // namespace ml_kem
  } // namespace pqc

} // namespace icicle