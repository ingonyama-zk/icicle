#include "icicle/mlkem/mlkem.h"
#include "icicle/hash/hash.h"
#include "icicle/backend/backend.h"
#include "icicle/backend/mlkem_backend.h"
#include "icicle/errors.h"
#include "icicle/dispatcher.h"
#include <random>

namespace mlkem {

// Define ML-KEM dispatcher
ICICLE_DISPATCHER_INST(MLKEMDispatcher, mlkem_impl, MLKEMImpl);

MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> keygen() {
    try {
        // ML-KEM parameters
        constexpr size_t n = 256;  // dimension of the lattice
        constexpr size_t k = 2;    // number of rows/columns in matrix A
        constexpr size_t q = 3329; // modulus
        constexpr size_t eta = 2;  // noise parameter
        
        // Get backend context
        auto ctx = backend::get_context();
        if (!ctx) {
            return MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>>::error(
                MLKEMError::BackendError, "Failed to get backend context");
        }
        
        // Initialize vectors for matrix A, secret key s, noise e, and result t
        std::vector<std::vector<uint8_t>> A(k, std::vector<uint8_t>(k * n));
        std::vector<uint8_t> s(k * n);
        std::vector<uint8_t> e(k * n);
        std::vector<uint8_t> t(k * n);
        
        // Execute ML-KEM key generation using dispatcher
        auto result = MLKEMDispatcher::execute(
            k, n, q, eta, A, s, e, t);
            
        if (result != eIcicleError::SUCCESS) {
            return MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>>::error(
                MLKEMError::BackendError, "Failed to execute ML-KEM key generation");
        }
        
        // Pack public key (A, t) and secret key s
        std::vector<uint8_t> public_key;
        public_key.reserve(k * k * n + k * n); // Size for A and t
        
        // Pack matrix A
        for (const auto& row : A) {
            public_key.insert(public_key.end(), row.begin(), row.end());
        }
        
        // Pack vector t
        public_key.insert(public_key.end(), t.begin(), t.end());
        
        return MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>>::success(
            std::make_pair(public_key, s));
    } catch (const std::exception& e) {
        return MLKEMResult<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>>::error(
            MLKEMError::InternalError, std::string("Unexpected error: ") + e.what());
    }
}

} // namespace mlkem 