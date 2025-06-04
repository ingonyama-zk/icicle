#pragma once

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {
  namespace pqc {
    namespace ml_kem {

      // TODO: maybe move SecuirityCategory to the config with a default value?

      // === Byte sizes ===
      constexpr size_t ENTROPY_BYTES = 64;
      constexpr size_t MESSAGE_BYTES = 32;

      struct Kyber512Params {
        static constexpr size_t PUBLIC_KEY_BYTES    = 800;
        static constexpr size_t SECRET_KEY_BYTES    = 1632;
        static constexpr size_t CIPHERTEXT_BYTES    = 768;
        static constexpr size_t SHARED_SECRET_BYTES = 32;
        static constexpr uint8_t K                  = 2;
        static constexpr uint8_t ETA1               = 3;
        static constexpr uint8_t ETA2               = 2;
        static constexpr uint8_t DU                 = 10;
        static constexpr uint8_t DV                 = 4;
      };

      struct Kyber768Params {
        static constexpr size_t PUBLIC_KEY_BYTES    = 1184;
        static constexpr size_t SECRET_KEY_BYTES    = 2400;
        static constexpr size_t CIPHERTEXT_BYTES    = 1088;
        static constexpr size_t SHARED_SECRET_BYTES = 32;
        static constexpr uint8_t K                  = 3;
        static constexpr uint8_t ETA1               = 2;
        static constexpr uint8_t ETA2               = 2;
        static constexpr uint8_t DU                 = 10;
        static constexpr uint8_t DV                 = 4;
      };

      struct Kyber1024Params {
        static constexpr size_t PUBLIC_KEY_BYTES    = 1568;
        static constexpr size_t SECRET_KEY_BYTES    = 3168;
        static constexpr size_t CIPHERTEXT_BYTES    = 1568;
        static constexpr size_t SHARED_SECRET_BYTES = 32;
        static constexpr uint8_t K                  = 4;
        static constexpr uint8_t ETA1               = 2;
        static constexpr uint8_t ETA2               = 2;
        static constexpr uint8_t DU                 = 11;
        static constexpr uint8_t DV                 = 5;
      };

      /// @brief Security category = parameter set = NIST level
      enum class SecurityCategory {
        KYBER_512 = 0,  // NIST Level 1
        KYBER_768 = 1,  // NIST Level 3
        KYBER_1024 = 2, // NIST Level 5
      };

      /// @brief Configuration for batch ML-KEM operations
      struct MlKemConfig {
        icicleStreamHandle stream = nullptr;
        bool is_async = false;

        // Host/device location hints for each buffer type
        bool messages_on_device = false;
        bool entropy_on_device = false;
        bool public_keys_on_device = false;
        bool secret_keys_on_device = false;
        bool ciphertexts_on_device = false;
        bool shared_secrets_on_device = false;

        int batch_size = 1;

        ConfigExtension* ext = nullptr; // Optional backend-specific settings
      };

      /**
       * @brief Generate a batch of ML-KEM keypairs.
       *
       * @param category      Security level (KYBER_512/768/1024)
       * @param entropy       Input buffer [batch_size × 64] bytes (uniform randomness)
       * @param config        Execution + memory config
       * @param public_keys   Output buffer [batch_size × PUBLIC_KEY_BYTES]
       * @param secret_keys   Output buffer [batch_size × SECRET_KEY_BYTES]
       */
      eIcicleError keygen512(
        const std::byte* entropy,
        MlKemConfig config,
        std::byte* public_keys,
        std::byte* secret_keys);

      /**
       * @brief Perform encapsulation for a batch of public keys.
       *
       * @param category        Security level (KYBER_512/768/1024)
       * @param public_keys     Input buffer [batch_size × PUBLIC_KEY_BYTES]
       * @param config          Execution + memory config
       * @param ciphertext      Output buffer [batch_size × CIPHERTEXT_BYTES]
       * @param shared_secrets  Output buffer [batch_size × SHARED_SECRET_BYTES]
       */
      eIcicleError encapsulate512(
        const std::byte* message,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets);

      /**
       * @brief Perform decapsulation for a batch of ciphertexts.
       *
       * @param category        Security level (KYBER_512/768/1024)
       * @param secret_keys     Input buffer [batch_size × SECRET_KEY_BYTES]
       * @param ciphertext      Input buffer [batch_size × CIPHERTEXT_BYTES]
       * @param config          Execution + memory config
       * @param shared_secrets  Output buffer [batch_size × SHARED_SECRET_BYTES]
       */
      eIcicleError decapsulate512(
        const std::byte* secret_keys,
        const std::byte* ciphertext,
        MlKemConfig config,
        std::byte* shared_secrets);

    } // namespace ml_kem
  } // namespace pqc
} // namespace icicle