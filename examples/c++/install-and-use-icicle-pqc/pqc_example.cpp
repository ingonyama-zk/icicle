#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>

// ICICLE PQC headers
#include "icicle/pqc/ml_kem.h"
#include "icicle/runtime.h"
#include "icicle/device.h"
#include "icicle/errors.h"

// Generate random bytes
void generate_random_bytes(std::byte* buffer, size_t size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 255);
    
    for (size_t i = 0; i < size; ++i) {
        buffer[i] = static_cast<std::byte>(dis(gen));
    }
}

// Benchmark ML-KEM operations for a specific parameter set
template<typename Params>
void benchmark_ml_kem(const std::string& param_name, int batch_size) {
    std::cout << "\n=== " << param_name << " Benchmark (batch_size=" << batch_size << ") ===" << std::endl;
    
    // Allocate buffers for batch operations
    std::vector<std::byte> entropy(icicle::pqc::ml_kem::ENTROPY_BYTES * batch_size);
    std::vector<std::byte> message(icicle::pqc::ml_kem::MESSAGE_BYTES * batch_size);
    std::vector<std::byte> public_key(Params::PUBLIC_KEY_BYTES * batch_size);
    std::vector<std::byte> secret_key(Params::SECRET_KEY_BYTES * batch_size);
    std::vector<std::byte> ciphertext(Params::CIPHERTEXT_BYTES * batch_size);
    std::vector<std::byte> shared_secret_enc(Params::SHARED_SECRET_BYTES * batch_size);
    std::vector<std::byte> shared_secret_dec(Params::SHARED_SECRET_BYTES * batch_size);
    
    // Generate random data for all batch items
    generate_random_bytes(entropy.data(), entropy.size());
    generate_random_bytes(message.data(), message.size());
    
    icicle::pqc::ml_kem::MlKemConfig config = {};
    config.batch_size = batch_size;
    
    // Benchmark key generation
    auto start = std::chrono::high_resolution_clock::now();
    auto keygen_result = icicle::pqc::ml_kem::keygen<Params>(
        entropy.data(), config, public_key.data(), secret_key.data()
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto keygen_total_ms = std::chrono::duration<float, std::milli>(end - start).count();
    auto keygen_avg_ms = keygen_total_ms / batch_size;
    
    if (keygen_result != eIcicleError::SUCCESS) {
        std::cout << "Key generation failed!" << std::endl;
        return;
    }
    
    // Benchmark encapsulation
    start = std::chrono::high_resolution_clock::now();
    auto encaps_result = icicle::pqc::ml_kem::encapsulate<Params>(
        message.data(), public_key.data(), config,
        ciphertext.data(), shared_secret_enc.data()
    );
    end = std::chrono::high_resolution_clock::now();
    auto encaps_total_ms = std::chrono::duration<float, std::milli>(end - start).count();
    auto encaps_avg_ms = encaps_total_ms / batch_size;
    
    if (encaps_result != eIcicleError::SUCCESS) {
        std::cout << "Encapsulation failed!" << std::endl;
        return;
    }
    
    // Benchmark decapsulation
    start = std::chrono::high_resolution_clock::now();
    auto decaps_result = icicle::pqc::ml_kem::decapsulate<Params>(
        secret_key.data(), ciphertext.data(), config,
        shared_secret_dec.data()
    );
    end = std::chrono::high_resolution_clock::now();
    auto decaps_total_ms = std::chrono::duration<float, std::milli>(end - start).count();
    auto decaps_avg_ms = decaps_total_ms / batch_size;
    
    if (decaps_result != eIcicleError::SUCCESS) {
        std::cout << "Decapsulation failed!" << std::endl;
        return;
    }
    
    // Verify shared secrets match for all batch items
    bool all_secrets_match = true;
    for (int i = 0; i < batch_size; ++i) {
        bool secrets_match = (0 == memcmp(
            shared_secret_enc.data() + i * Params::SHARED_SECRET_BYTES,
            shared_secret_dec.data() + i * Params::SHARED_SECRET_BYTES,
            Params::SHARED_SECRET_BYTES
        ));
        if (!secrets_match) {
            all_secrets_match = false;
            break;
        }
    }
    
    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Keygen (avg): " << std::setw(8) << keygen_avg_ms << " ms  (total: " << keygen_total_ms << " ms)" << std::endl;
    std::cout << "Encaps (avg): " << std::setw(8) << encaps_avg_ms << " ms  (total: " << encaps_total_ms << " ms)" << std::endl;
    std::cout << "Decaps (avg): " << std::setw(8) << decaps_avg_ms << " ms  (total: " << decaps_total_ms << " ms)" << std::endl;
    std::cout << "Verification: " << (all_secrets_match ? "✅ PASS" : "❌ FAIL") << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int batch_size = 1;  // default batch size
    
    if (argc > 1) {
        try {
            batch_size = std::stoi(argv[1]);
            if (batch_size <= 0) {
                std::cout << "Error: Batch size must be positive" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid batch size argument" << std::endl;
            std::cout << "Usage: " << argv[0] << " [batch_size]" << std::endl;
            return 1;
        }
    }
    
    std::cout << "ICICLE ML-KEM Benchmark\n" << std::endl;
    
    // Initialize device
    // auto load_result = icicle_load_backend_from_env_or_default();
    // if (load_result == eIcicleError::SUCCESS) {
    //     if (icicle_is_device_available("CUDA-PQC") == eIcicleError::SUCCESS) {
    //         icicle::Device dev = {"CUDA-PQC", 0};
    //         icicle_set_device(dev);
    //         std::cout << "Using CUDA-PQC device" << std::endl;
    //     } else {
    //         std::cout << "Using CPU device" << std::endl;
    //     }
    // } else {
    icicle::Device dev = {"CUDA-PQC", 0};
    auto err = icicle_set_device(dev);
    dev = icicle::DeviceAPI::get_thread_local_device();
    std::cout << "Using device: " << dev << std::endl;
    //     std::cout << get_error_string(err) << std::endl;
    // }
    
    // Benchmark all parameter sets
    benchmark_ml_kem<icicle::pqc::ml_kem::Kyber512Params>("ML-KEM-512", batch_size);
    benchmark_ml_kem<icicle::pqc::ml_kem::Kyber768Params>("ML-KEM-768", batch_size);
    benchmark_ml_kem<icicle::pqc::ml_kem::Kyber1024Params>("ML-KEM-1024", batch_size);
    
    return 0;
} 