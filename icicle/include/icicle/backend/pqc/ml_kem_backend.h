#pragma once

#include "icicle/pqc/ml_kem.h"
#include <functional>

namespace icicle {
  namespace pqc {
    namespace ml_kem {
      using MlKemKeygenImpl = std::function<eIcicleError(
        const Device& device,
        const std::byte* entropy,
        MlKemConfig config,
        std::byte* public_keys,
        std::byte* secret_keys)>;

      using MlKemEncapsulateImpl = std::function<eIcicleError(
        const Device& device,
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)>;

      using MlKemDecapsulateImpl = std::function<eIcicleError(
        const Device& device,
        const std::byte* secret_keys,
        const std::byte* ciphertext,
        MlKemConfig config,
        std::byte* shared_secrets)>;

      void register_ml_kem_keygen512(const std::string& deviceType, MlKemKeygenImpl impl);
      void register_ml_kem_encaps512(const std::string& deviceType, MlKemEncapsulateImpl impl);
      void register_ml_kem_decaps512(const std::string& deviceType, MlKemDecapsulateImpl impl);

      void register_ml_kem_keygen768(const std::string& deviceType, MlKemKeygenImpl impl);
      void register_ml_kem_encaps768(const std::string& deviceType, MlKemEncapsulateImpl impl);
      void register_ml_kem_decaps768(const std::string& deviceType, MlKemDecapsulateImpl impl);

      void register_ml_kem_keygen1024(const std::string& deviceType, MlKemKeygenImpl impl);
      void register_ml_kem_encaps1024(const std::string& deviceType, MlKemEncapsulateImpl impl);
      void register_ml_kem_decaps1024(const std::string& deviceType, MlKemDecapsulateImpl impl);

#define REGISTER_ML_KEM512_BACKEND(DEVICE_TYPE, KEYGEN, ENCAPS, DECAPS)                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ml_kem512) = []() -> bool {                                                                \
      register_ml_kem_keygen512(DEVICE_TYPE, KEYGEN<Kyber512Params>);                                                  \
      register_ml_kem_encaps512(DEVICE_TYPE, ENCAPS<Kyber512Params>);                                                  \
      register_ml_kem_decaps512(DEVICE_TYPE, DECAPS<Kyber512Params>);                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#define REGISTER_ML_KEM768_BACKEND(DEVICE_TYPE, KEYGEN, ENCAPS, DECAPS)                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ml_kem768) = []() -> bool {                                                                \
      register_ml_kem_keygen768(DEVICE_TYPE, KEYGEN<Kyber768Params>);                                                  \
      register_ml_kem_encaps768(DEVICE_TYPE, ENCAPS<Kyber768Params>);                                                  \
      register_ml_kem_decaps768(DEVICE_TYPE, DECAPS<Kyber768Params>);                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#define REGISTER_ML_KEM1024_BACKEND(DEVICE_TYPE, KEYGEN, ENCAPS, DECAPS)                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ml_kem1024) = []() -> bool {                                                               \
      register_ml_kem_keygen1024(DEVICE_TYPE, KEYGEN<Kyber1024Params>);                                                \
      register_ml_kem_encaps1024(DEVICE_TYPE, ENCAPS<Kyber1024Params>);                                                \
      register_ml_kem_decaps1024(DEVICE_TYPE, DECAPS<Kyber1024Params>);                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
    } // namespace ml_kem
  } // namespace pqc
} // namespace icicle