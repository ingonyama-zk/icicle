#include "test_base.h"
#include "icicle/mlkem/mlkem.h"
#include <gtest/gtest.h>

class TestMLKEM : public IcicleTestBase {
protected:
    void SetUp() override {
        IcicleTestBase::SetUp();
    }
};

TEST_F(TestMLKEM, KeyGenBasic) {
    // Test basic key generation
    auto [public_key, secret_key] = mlkem::keygen();
    
    // Check sizes
    constexpr size_t n = 256;  // dimension of the lattice
    constexpr size_t k = 2;    // number of rows/columns in matrix A
    constexpr size_t q = 3329; // modulus
    
    // Public key should contain matrix A (k x k x n) and vector t (k x n)
    EXPECT_EQ(public_key.size(), k * k * n + k * n);
    
    // Secret key should contain vector s (k x n)
    EXPECT_EQ(secret_key.size(), k * n);
    
    // Check that all values are within valid range
    for (uint8_t val : public_key) {
        EXPECT_LT(val, q);
    }
    
    for (uint8_t val : secret_key) {
        EXPECT_LT(val, 2 * 2); // eta = 2, so values should be in [0, 2*eta]
    }
}

TEST_F(TestMLKEM, KeyGenConsistency) {
    // Test that key generation produces consistent results
    auto [pk1, sk1] = mlkem::keygen();
    auto [pk2, sk2] = mlkem::keygen();
    
    // Different runs should produce different keys
    EXPECT_NE(pk1, pk2);
    EXPECT_NE(sk1, sk2);
}

TEST_F(TestMLKEM, KeyGenPerformance) {
    // Test performance of key generation
    const int num_runs = 100;
    START_TIMER(keygen);
    
    for (int i = 0; i < num_runs; i++) {
        auto [public_key, secret_key] = mlkem::keygen();
    }
    
    END_TIMER_AVERAGE(keygen, "ML-KEM KeyGen average time", true, num_runs);
}

TEST_F(TestMLKEM, KeyGenStructure) {
    // Test the structure of generated keys
    auto [public_key, secret_key] = mlkem::keygen();
    
    constexpr size_t n = 256;  // dimension of the lattice
    constexpr size_t k = 2;    // number of rows/columns in matrix A
    
    // Check matrix A structure in public key
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < k; j++) {
            for (size_t l = 0; l < n; l++) {
                size_t idx = i * k * n + j * n + l;
                EXPECT_LT(public_key[idx], 3329); // Check modulus
            }
        }
    }
    
    // Check vector t structure in public key
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            size_t idx = k * k * n + i * n + j;
            EXPECT_LT(public_key[idx], 3329); // Check modulus
        }
    }
    
    // Check secret key structure
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            size_t idx = i * n + j;
            EXPECT_LT(secret_key[idx], 4); // Check noise bound (2 * eta)
        }
    }
} 