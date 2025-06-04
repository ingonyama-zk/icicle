#include "icicle/pqc/ml_kem.h"

#include "icicle/errors.h"
#include "icicle/dispatcher.h"
#include "icicle/backend/pqc/ml_kem_backend.h"

namespace icicle {
  namespace pqc {
    namespace ml_kem {

      // ICICLE_DISPATCHER_INST(MlKemKeygenDispatcher1024, ml_kem_keygen, MlKemKeygenImpl<Kyber1024Params>);
      // ICICLE_DISPATCHER_INST(MlKemEncapsulateDispatcher1024, ml_kem_encaps, MlKemEncapsulateImpl<Kyber1024Params>);
      // ICICLE_DISPATCHER_INST(MlKemDecapsulateDispatcher1024, ml_kem_decaps, MlKemDecapsulateImpl<Kyber1024Params>);

      // ICICLE_DISPATCHER_INST(MlKemKeygenDispatcher768, ml_kem_keygen, MlKemKeygenImpl<Kyber768Params>);
      // ICICLE_DISPATCHER_INST(MlKemEncapsulateDispatcher768, ml_kem_encaps, MlKemEncapsulateImpl<Kyber768Params>);
      // ICICLE_DISPATCHER_INST(MlKemDecapsulateDispatcher768, ml_kem_decaps, MlKemDecapsulateImpl<Kyber768Params>);

      ICICLE_DISPATCHER_INST(MlKemKeygenDispatcher512, ml_kem_keygen, MlKemKeygenImpl<Kyber512Params>);
      ICICLE_DISPATCHER_INST(MlKemEncapsulateDispatcher512, ml_kem_encaps, MlKemEncapsulateImpl<Kyber512Params>);
      ICICLE_DISPATCHER_INST(MlKemDecapsulateDispatcher512, ml_kem_decaps, MlKemDecapsulateImpl<Kyber512Params>);

      eIcicleError keygen512(
        const std::byte* entropy,
        MlKemConfig config,
        std::byte* public_keys,
        std::byte* secret_keys)
      {
        return MlKemKeygenDispatcher512::execute(entropy, config, public_keys, secret_keys);
      }

      eIcicleError encapsulate512(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return MlKemEncapsulateDispatcher512::execute(message, public_keys, config, ciphertext, shared_secrets);
      }

      eIcicleError decapsulate512(
        const std::byte* secret_keys,
        const std::byte* ciphertext,
        MlKemConfig config,
        std::byte* shared_secrets)
      {
        return MlKemDecapsulateDispatcher512::execute(secret_keys, ciphertext, config, shared_secrets);
      }

      // TODO: C FFI

    } // namespace ml_kem
  } // namespace pqc

} // namespace icicle