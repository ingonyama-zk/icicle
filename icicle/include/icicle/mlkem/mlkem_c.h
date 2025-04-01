#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ML-KEM key pair structure
typedef struct {
    uint8_t* public_key;
    size_t public_key_len;
    uint8_t* secret_key;
    size_t secret_key_len;
} MLKEMKeyPair;

// ML-KEM ciphertext structure
typedef struct {
    uint8_t* ciphertext;
    size_t ciphertext_len;
    uint8_t* shared_secret;
    size_t shared_secret_len;
} MLKEMCiphertext;

// Initialize the ML-KEM backend
// Returns 0 on success, non-zero on failure
uint32_t mlkem_init(uint32_t backend_type);

// Generate a new ML-KEM key pair
// Returns a pointer to MLKEMKeyPair on success, NULL on failure
MLKEMKeyPair* mlkem_keygen(void);

// Encapsulate a shared secret using a public key
// Returns a pointer to MLKEMCiphertext on success, NULL on failure
MLKEMCiphertext* mlkem_encaps(const uint8_t* public_key, size_t public_key_len);

// Decapsulate a shared secret using a secret key
// Returns a pointer to the shared secret on success, NULL on failure
uint8_t* mlkem_decaps(const uint8_t* ciphertext, size_t ciphertext_len,
                     const uint8_t* secret_key, size_t secret_key_len);

// Free a ML-KEM key pair
void mlkem_free_keypair(MLKEMKeyPair* keypair);

// Free a ML-KEM ciphertext
void mlkem_free_ciphertext(MLKEMCiphertext* ciphertext);

// Free a shared secret
void mlkem_free_shared_secret(uint8_t* shared_secret);

#ifdef __cplusplus
}
#endif 