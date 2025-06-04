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

#define REGISTER_ML_KEM_BACKEND(DEVICE_TYPE, KEYGEN, ENCAPS, DECAPS)                                                   \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ml_kem) = []() -> bool {                                                   \
      register_ml_kem_keygen512(DEVICE_TYPE, KEYGEN);                                                                     \
      register_ml_kem_encaps512(DEVICE_TYPE, ENCAPS);                                                                     \
      register_ml_kem_decaps512(DEVICE_TYPE, DECAPS);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
    } // namespace ml_kem
  } // namespace pqc
} // namespace icicle