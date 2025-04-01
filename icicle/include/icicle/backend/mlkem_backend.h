#pragma once

#include "icicle/backend/backend.h"
#include "icicle/errors.h"
#include <vector>
#include <memory>

namespace icicle {

// ML-KEM backend interface
class MLKEMBackend {
public:
    virtual ~MLKEMBackend() = default;
    
    // Generate matrix A
    virtual eIcicleError generate_matrix_A(
        const Device& device,
        uint32_t k,
        uint32_t n,
        uint32_t q,
        std::vector<std::vector<uint8_t>>& A) = 0;
    
    // Generate secret key s
    virtual eIcicleError generate_secret_key(
        const Device& device,
        uint32_t k,
        uint32_t n,
        uint32_t eta,
        std::vector<uint8_t>& s) = 0;
    
    // Generate noise e
    virtual eIcicleError generate_noise(
        const Device& device,
        uint32_t k,
        uint32_t n,
        uint32_t eta,
        std::vector<uint8_t>& e) = 0;
    
    // Matrix-vector multiplication and addition
    virtual eIcicleError matrix_vector_multiply_add(
        const Device& device,
        const std::vector<std::vector<uint8_t>>& A,
        const std::vector<uint8_t>& s,
        const std::vector<uint8_t>& e,
        uint32_t q,
        std::vector<uint8_t>& t) = 0;
};

// ML-KEM backend implementation type
using MLKEMImpl = std::function<eIcicleError(
    const Device&,
    uint32_t k,
    uint32_t n,
    uint32_t q,
    uint32_t eta,
    std::vector<std::vector<uint8_t>>& A,
    std::vector<uint8_t>& s,
    std::vector<uint8_t>& e,
    std::vector<uint8_t>& t)>;

// Registration function
void register_mlkem_impl(const std::string& deviceType, MLKEMImpl impl);

// Registration macro
#define REGISTER_MLKEM_BACKEND(DEVICE_TYPE, FUNC) \
    void register_mlkem_##DEVICE_TYPE##_backend() { \
        register_mlkem_impl(DEVICE_TYPE, FUNC); \
    }

} // namespace icicle 