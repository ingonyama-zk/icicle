#pragma once

#include "icicle/errors.h"
#include "icicle/fields/stark_fields/koalabear.h"
#include "icicle/backend/backend.h"
#include <cstdint>
#include <vector>
#include <string>
#include <variant>
#include <memory>

namespace mlkem {

// ML-KEM error types
enum class MLKEMError {
    Success,
    BackendError,
    InvalidInput,
    InternalError
};

// ML-KEM result type
template<typename T>
class MLKEMResult {
public:
    static MLKEMResult<T> success(T value) {
        return MLKEMResult<T>(std::move(value));
    }
    
    static MLKEMResult<T> error(MLKEMError error, std::string message) {
        return MLKEMResult<T>(error, std::move(message));
    }
    
    bool is_success() const {
        return std::holds_alternative<T>(result_);
    }
    
    const T& get_value() const {
        return std::get<T>(result_);
    }
    
    MLKEMError get_error() const {
        return std::get<std::pair<MLKEMError, std::string>>(result_).first;
    }
    
    const std::string& get_error_message() const {
        return std::get<std::pair<MLKEMError, std::string>>(result_).second;
    }

private:
    MLKEMResult(T value) : result_(std::move(value)) {}
    MLKEMResult(MLKEMError error, std::string message)
        : result_(std::make_pair(error, std::move(message))) {}
    
    std::variant<T, std::pair<MLKEMError, std::string>> result_;
};

// Initialize the ML-KEM backend
MLKEMResult<void> init(backend::BackendType backend_type);

// Get the current ML-KEM backend
std::shared_ptr<backend::Backend> get_backend();

// ML-KEM key generation function
// Returns a MLKEMResult containing either:
// - A pair of (public_key, secret_key) on success
// - An error code and message on failure
MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> keygen();

// ML-KEM encapsulation function
// Returns a MLKEMResult containing either:
// - A pair of (ciphertext, shared_secret) on success
// - An error code and message on failure
MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> encaps(
    const std::vector<uint8_t>& public_key);

// ML-KEM decapsulation function
// Returns a MLKEMResult containing either:
// - The shared secret on success
// - An error code and message on failure
MLKEMResult<std::vector<uint8_t>> decaps(
    const std::vector<uint8_t>& ciphertext,
    const std::vector<uint8_t>& secret_key);

} // namespace mlkem 