#ifndef _ICICLE_ML_KEM_H
#define _ICICLE_ML_KEM_H

#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MlKemConfig MlKemConfig;

// ========================
// ML-KEM 512-bit functions
// ========================

int icicle_ml_kem_keygen512(
    const uint8_t* entropy,
    const MlKemConfig* config,
    uint8_t* public_keys,
    uint8_t* secret_keys
);

int icicle_ml_kem_encapsulate512(
    const uint8_t* message,
    const uint8_t* public_keys,
    const MlKemConfig* config,
    uint8_t* ciphertexts,
    uint8_t* shared_secrets
);

int icicle_ml_kem_decapsulate512(
    const uint8_t* secret_keys,
    const uint8_t* ciphertexts,
    const MlKemConfig* config,
    uint8_t* shared_secrets
);

// ========================
// ML-KEM 768-bit functions
// ========================

int icicle_ml_kem_keygen768(
    const uint8_t* entropy,
    const MlKemConfig* config,
    uint8_t* public_keys,
    uint8_t* secret_keys
);

int icicle_ml_kem_encapsulate768(
    const uint8_t* message,
    const uint8_t* public_keys,
    const MlKemConfig* config,
    uint8_t* ciphertexts,
    uint8_t* shared_secrets
);

int icicle_ml_kem_decapsulate768(
    const uint8_t* secret_keys,
    const uint8_t* ciphertexts,
    const MlKemConfig* config,
    uint8_t* shared_secrets
);

// =========================
// ML-KEM 1024-bit functions
// =========================

int icicle_ml_kem_keygen1024(
    const uint8_t* entropy,
    const MlKemConfig* config,
    uint8_t* public_keys,
    uint8_t* secret_keys
);

int icicle_ml_kem_encapsulate1024(
    const uint8_t* message,
    const uint8_t* public_keys,
    const MlKemConfig* config,
    uint8_t* ciphertexts,
    uint8_t* shared_secrets
);

int icicle_ml_kem_decapsulate1024(
    const uint8_t* secret_keys,
    const uint8_t* ciphertexts,
    const MlKemConfig* config,
    uint8_t* shared_secrets
);

#ifdef __cplusplus
}
#endif

#endif // _ICICLE_ML_KEM_H
