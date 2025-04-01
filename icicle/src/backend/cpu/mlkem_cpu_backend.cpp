#include "icicle/backend/mlkem_backend.h"
#include "icicle/device.h"
#include "icicle/errors.h"
#include <random>

namespace icicle {

class MLKEMCPUBackend : public MLKEMBackend {
public:
    eIcicleError generate_matrix_A(
        const Device& device,
        uint32_t k,
        uint32_t n,
        uint32_t q,
        std::vector<std::vector<uint8_t>>& A) override {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, q - 1);
        
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < k; j++) {
                for (size_t l = 0; l < n; l++) {
                    A[i][j * n + l] = static_cast<uint8_t>(dis(gen));
                }
            }
        }
        
        return eIcicleError::SUCCESS;
    }
    
    eIcicleError generate_secret_key(
        const Device& device,
        uint32_t k,
        uint32_t n,
        uint32_t eta,
        std::vector<uint8_t>& s) override {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, 2 * eta);
        
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < n; j++) {
                s[i * n + j] = static_cast<uint8_t>(dis(gen));
            }
        }
        
        return eIcicleError::SUCCESS;
    }
    
    eIcicleError generate_noise(
        const Device& device,
        uint32_t k,
        uint32_t n,
        uint32_t eta,
        std::vector<uint8_t>& e) override {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, 2 * eta);
        
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < n; j++) {
                e[i * n + j] = static_cast<uint8_t>(dis(gen));
            }
        }
        
        return eIcicleError::SUCCESS;
    }
    
    eIcicleError matrix_vector_multiply_add(
        const Device& device,
        const std::vector<std::vector<uint8_t>>& A,
        const std::vector<uint8_t>& s,
        const std::vector<uint8_t>& e,
        uint32_t q,
        std::vector<uint8_t>& t) override {
        
        // Matrix-vector multiplication
        for (size_t i = 0; i < A.size(); i++) {
            for (size_t j = 0; j < s.size() / A.size(); j++) {
                uint32_t sum = 0;
                for (size_t k = 0; k < A.size(); k++) {
                    sum += static_cast<uint32_t>(A[i][k * s.size() / A.size() + j]) *
                           static_cast<uint32_t>(s[k * s.size() / A.size() + j]);
                }
                // Add noise and reduce modulo q
                sum = (sum + static_cast<uint32_t>(e[i * s.size() / A.size() + j])) % q;
                t[i * s.size() / A.size() + j] = static_cast<uint8_t>(sum);
            }
        }
        
        return eIcicleError::SUCCESS;
    }
};

// CPU backend implementation function
eIcicleError cpu_mlkem_impl(
    const Device& device,
    uint32_t k,
    uint32_t n,
    uint32_t q,
    uint32_t eta,
    std::vector<std::vector<uint8_t>>& A,
    std::vector<uint8_t>& s,
    std::vector<uint8_t>& e,
    std::vector<uint8_t>& t) {
    
    auto cpu_backend = std::make_shared<MLKEMCPUBackend>();
    
    // Generate matrix A
    auto result = cpu_backend->generate_matrix_A(device, k, n, q, A);
    if (result != eIcicleError::SUCCESS) return result;
    
    // Generate secret key s
    result = cpu_backend->generate_secret_key(device, k, n, eta, s);
    if (result != eIcicleError::SUCCESS) return result;
    
    // Generate noise e
    result = cpu_backend->generate_noise(device, k, n, eta, e);
    if (result != eIcicleError::SUCCESS) return result;
    
    // Matrix-vector multiplication and addition
    return cpu_backend->matrix_vector_multiply_add(device, A, s, e, q, t);
}

// Register CPU backend implementation
REGISTER_MLKEM_BACKEND("CPU", cpu_mlkem_impl)

} // namespace icicle 