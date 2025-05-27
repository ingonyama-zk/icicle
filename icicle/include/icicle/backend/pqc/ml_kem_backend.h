#pragma once

#include "icicle/pqc/ml_kem.h"
#include <functional>

namespace icicle {
  namespace pqc {
    namespace ml_kem {
      using MlKemKeygenImpl = std::function<eIcicleError(
        const Device& device,
        SecurityCategory category,
        const std::byte* entropy,
        MlKemConfig config,
        std::byte* public_keys,
        std::byte* secret_keys)>;

      using MlKemEncapsulateImpl = std::function<eIcicleError(
        const Device& device,
        SecurityCategory category,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)>;
      using MlKemDecapsulateImpl = std::function<eIcicleError(
        const Device& device,
        SecurityCategory category,
        const std::byte* secret_keys,
        const std::byte* ciphertext,
        MlKemConfig config,
        std::byte* shared_secrets)>;

      void register_ml_kem_keygen(const std::string& deviceType, MlKemKeygenImpl impl);
      void register_ml_kem_encapsulate(const std::string& deviceType, MlKemEncapsulateImpl impl);
      void register_ml_kem_decapsulate(const std::string& deviceType, MlKemDecapsulateImpl impl);

#define REGISTER_ML_KEM_BACKEND(DEVICE_TYPE, KEYGEN, ENCAP, DECAP)                                                     \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_balanced_recomposition) = []() -> bool {                                                   \
      register_ml_kem_keygen(DEVICE_TYPE, KEYGEN);                                                                     \
      register_ml_kem_encapsulate(DEVICE_TYPE, ENCAP);                                                                 \
      register_ml_kem_decapsulate(DEVICE_TYPE, DECAP);                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
    } // namespace ml_kem
  } // namespace pqc
} // namespace icicle