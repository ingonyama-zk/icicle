# ML-KEM – Post-Quantum Key Encapsulation (Kyber)

## Overview

**ML-KEM** (Module-Lattice Key Encapsulation Mechanism) – is a lattice-based key-encapsulation protocol selected by NIST for post-quantum cryptography ([FIPS 203](https://csrc.nist.gov/pubs/fips/203/final)).  
ML-KEM provides three security categories that correspond to the Kyber512 (Level 1), Kyber768 (Level 3) and Kyber1024 (Level 5) parameter sets.

---

## Byte sizes

| Parameter set | Public key | Secret key | Cipher-text | Shared secret | Entropy bytes | Message bytes | Security category |
|--------------:|-----------:|-----------:|------------:|--------------:|--------------:|--------------:|-------------------|
| Kyber512      | 800 B      | 1632 B     | 768 B       | 32 B          | 64 B          | 32 B          | **Level 1** |
| Kyber768      | 1184 B     | 2400 B     | 1088 B      | 32 B          | 64 B          | 32 B          | **Level 3** |
| Kyber1024     | 1568 B     | 3168 B     | 1568 B      | 32 B          | 64 B          | 32 B          | **Level 5** |

---

## C++ API

All three operations are **templated** by the Kyber parameter set (`Kyber512Params`, `Kyber768Params`, `Kyber1024Params`).  The templates are explicitly instantiated by Icicle, so you only have to include the header and link against `icicle_pqc`.

### `MlKemConfig` struct

```cpp
struct MlKemConfig {
  icicleStreamHandle stream = nullptr; // Optional async stream
  bool is_async                   = false; // If true – return immediately and synchronize later

  // Location hints – set to `true` if the corresponding buffer already resides on the device
  bool messages_on_device         = false;
  bool entropy_on_device          = false;
  bool public_keys_on_device      = false;
  bool secret_keys_on_device      = false;
  bool ciphertexts_on_device      = false;
  bool shared_secrets_on_device   = false;

  int  batch_size = 1;                     // Number of independent KEMs processed in parallel

  ConfigExtension* ext = nullptr;          // Backend-specific tuning knobs (optional)
};
```

### Key pair generation

```cpp
template <typename Params /* Kyber512Params | Kyber768Params | Kyber1024Params */>
eIcicleError keygen(
    const std::byte* entropy,      // [batch_size × ENTROPY_BYTES]
    MlKemConfig        config,
    std::byte*         public_keys, // [batch_size × Params::PUBLIC_KEY_BYTES]
    std::byte*         secret_keys  // [batch_size × Params::SECRET_KEY_BYTES]
);
```

### Encapsulation

```cpp
template <typename Params>
eIcicleError encapsulate(
    const std::byte* message,       // [batch_size × MESSAGE_BYTES] arbitrary plaintext
    const std::byte* public_keys,   // [batch_size × Params::PUBLIC_KEY_BYTES]
    MlKemConfig        config,
    std::byte*         ciphertexts, // [batch_size × Params::CIPHERTEXT_BYTES]
    std::byte*         shared_secrets // [batch_size × Params::SHARED_SECRET_BYTES]
);
```

### Decapsulation

```cpp
template <typename Params>
eIcicleError decapsulate(
    const std::byte* secret_keys,   // [batch_size × Params::SECRET_KEY_BYTES]
    const std::byte* ciphertexts,   // [batch_size × Params::CIPHERTEXT_BYTES]
    MlKemConfig        config,
    std::byte*         shared_secrets // [batch_size × Params::SHARED_SECRET_BYTES]
);
```

---

## C API symbols

For integration with C or other FFI layers, Icicle exposes thin wrappers with the naming scheme `icicle_ml_kem_<op><level>`:

```c
// Level-1 (Kyber512)
eIcicleError icicle_ml_kem_keygen512(const uint8_t* entropy, const MlKemConfig* cfg,
                                     uint8_t* pk, uint8_t* sk);
eIcicleError icicle_ml_kem_encapsulate512(const uint8_t* msg, const uint8_t* pk,
                                          const MlKemConfig* cfg, uint8_t* ct, uint8_t* ss);
eIcicleError icicle_ml_kem_decapsulate512(const uint8_t* sk, const uint8_t* ct,
                                          const MlKemConfig* cfg, uint8_t* ss);
// … identical pattern for 768 / 1024
```

---

## Example: generate & encapsulate (Kyber768)

```cpp
#include "icicle/pqc/ml_kem.h"
using namespace icicle::pqc::ml_kem;

int main() {
const int batch_size = 1 << 12;
  // Config
  MlKemConfig config;
  config.batch_size = batch_size;

  // Allocate buffers
  auto entropy = this->random_entropy(batch_size * ENTROPY_BYTES);
  std::vector<std::byte> public_key(batch_size * TypeParam::PUBLIC_KEY_BYTES);
  std::vector<std::byte> secret_key(batch_size * TypeParam::SECRET_KEY_BYTES);
  std::vector<std::byte> ciphertext(batch_size * TypeParam::CIPHERTEXT_BYTES);
  std::vector<std::byte> shared_secret_enc(batch_size * TypeParam::SHARED_SECRET_BYTES);
  std::vector<std::byte> shared_secret_dec(batch_size * TypeParam::SHARED_SECRET_BYTES);

  auto message = this->random_entropy(batch_size * MESSAGE_BYTES);

  // Key generation
  auto err = keygen<TypeParam>(entropy.data(), config, public_key.data(), secret_key.data());

  // Encapsulation
  err = encapsulate<TypeParam>(message.data(), public_key.data(), config, ciphertext.data(), shared_secret_enc.data());

  // Decapsulation
  err = decapsulate<TypeParam>(secret_key.data(), ciphertext.data(), config, shared_secret_dec.data());
}
```

---

## FFI symbols

If you need to interface with C or another language, link to the symbols exported by `icicle_pqc`:

* `icicle_ml_kem_keygen512`, `icicle_ml_kem_encapsulate512`, `icicle_ml_kem_decapsulate512`
* `icicle_ml_kem_keygen768`, `icicle_ml_kem_encapsulate768`, `icicle_ml_kem_decapsulate768`
* `icicle_ml_kem_keygen1024`, `icicle_ml_kem_encapsulate1024`, `icicle_ml_kem_decapsulate1024`

---

## References  

• [NIST FIPS 203 – *Module-Lattice-Based Key-Encapsulation Mechanism Standard*](https://csrc.nist.gov/pubs/fips/203/final).
