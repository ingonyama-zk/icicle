#include "icicle/mlkem/mlkem.h"
#include <cstring>
#include <memory>

extern "C" {

uint32_t mlkem_init(uint32_t backend_type) {
    auto result = mlkem::init(static_cast<backend::BackendType>(backend_type));
    return result.is_success() ? 0 : 1;
}

MLKEMKeyPair* mlkem_keygen() {
    auto result = mlkem::keygen();
    if (!result.is_success()) {
        return nullptr;
    }

    auto [public_key, secret_key] = result.get_value();
    
    auto keypair = new MLKEMKeyPair;
    keypair->public_key_len = public_key.size();
    keypair->secret_key_len = secret_key.size();
    
    keypair->public_key = new uint8_t[keypair->public_key_len];
    keypair->secret_key = new uint8_t[keypair->secret_key_len];
    
    std::memcpy(keypair->public_key, public_key.data(), keypair->public_key_len);
    std::memcpy(keypair->secret_key, secret_key.data(), keypair->secret_key_len);
    
    return keypair;
}

MLKEMCiphertext* mlkem_encaps(const uint8_t* public_key, size_t public_key_len) {
    std::vector<uint8_t> pk(public_key, public_key + public_key_len);
    auto result = mlkem::encaps(pk);
    if (!result.is_success()) {
        return nullptr;
    }

    auto [ciphertext, shared_secret] = result.get_value();
    
    auto ct = new MLKEMCiphertext;
    ct->ciphertext_len = ciphertext.size();
    ct->shared_secret_len = shared_secret.size();
    
    ct->ciphertext = new uint8_t[ct->ciphertext_len];
    ct->shared_secret = new uint8_t[ct->shared_secret_len];
    
    std::memcpy(ct->ciphertext, ciphertext.data(), ct->ciphertext_len);
    std::memcpy(ct->shared_secret, shared_secret.data(), ct->shared_secret_len);
    
    return ct;
}

uint8_t* mlkem_decaps(const uint8_t* ciphertext, size_t ciphertext_len,
                     const uint8_t* secret_key, size_t secret_key_len) {
    std::vector<uint8_t> ct(ciphertext, ciphertext + ciphertext_len);
    std::vector<uint8_t> sk(secret_key, secret_key + secret_key_len);
    
    auto result = mlkem::decaps(ct, sk);
    if (!result.is_success()) {
        return nullptr;
    }

    auto shared_secret = result.get_value();
    auto ss = new uint8_t[shared_secret.size()];
    std::memcpy(ss, shared_secret.data(), shared_secret.size());
    
    return ss;
}

void mlkem_free_keypair(MLKEMKeyPair* keypair) {
    if (keypair) {
        delete[] keypair->public_key;
        delete[] keypair->secret_key;
        delete keypair;
    }
}

void mlkem_free_ciphertext(MLKEMCiphertext* ciphertext) {
    if (ciphertext) {
        delete[] ciphertext->ciphertext;
        delete[] ciphertext->shared_secret;
        delete ciphertext;
    }
}

void mlkem_free_shared_secret(uint8_t* shared_secret) {
    delete[] shared_secret;
}

} // extern "C" 