#include <cuda_runtime.h>
#include "ml_kem/ring/cuda_zq.cuh"

#include "ml_kem/cuda_ml_kem_kernels.cuh"

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <fstream>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "test_base.h"
#include "icicle/utils/rand_gen.h"

using namespace icicle;
using namespace icicle::pqc::ml_kem;

// static bool VERBOSE = true;
// static int ITERS = 1;

class MLKemKeygenTest : public IcicleTestBase
{
public:
  static const std::string TEST_DATA_PATH;
  static const size_t BATCH_SIZE;

  void SetUp() override
  {
    IcicleTestBase::SetUp();
    std::cout << "\nRunning tests with batch size: " << BATCH_SIZE << std::endl;
  }

  template <typename T>
  static void randomize(T* arr, uint64_t size)
  {
    // Fill the array with random values
    uint32_t* u32_arr = (uint32_t*)arr;
    for (int i = 0; i < (size * sizeof(T) / sizeof(uint32_t)); ++i) {
      u32_arr[i] = rand_uint_32b();
    }
  }

  static std::string voidPtrToHexString(const std::byte* byteData, size_t size)
  {
    std::ostringstream hexStream;
    for (size_t i = 0; i < size; ++i) {
      hexStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byteData[i]);
    }
    return hexStream.str();
  }
};

// Define the static constants
const std::string MLKemKeygenTest::TEST_DATA_PATH = "./icicle/backend/cuda_pqc/tests/ml_kem/test_data";
const size_t MLKemKeygenTest::BATCH_SIZE = 8192;
// const size_t MLKemKeygenTest::BATCH_SIZE = 256;

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen512)
{
  // Read test data files
  std::ifstream random_file(TEST_DATA_PATH + "/ml_kem_512_data/ml_kem_512_8192_random_bytes.txt", std::ios::binary);
  std::ifstream ek_file(TEST_DATA_PATH + "/ml_kem_512_data/ml_kem_512_8192_ek.txt", std::ios::binary);
  std::ifstream dk_file(TEST_DATA_PATH + "/ml_kem_512_data/ml_kem_512_8192_dk.txt", std::ios::binary);

  ASSERT_TRUE(random_file.is_open()) << "Failed to open random bytes file";
  ASSERT_TRUE(ek_file.is_open()) << "Failed to open encryption key file";
  ASSERT_TRUE(dk_file.is_open()) << "Failed to open decryption key file";

  // Read first key pair data
  std::vector<uint8_t> d_bytes(32);
  std::vector<uint8_t> z_bytes(32);
  random_file.read(reinterpret_cast<char*>(d_bytes.data()), 32);
  random_file.read(reinterpret_cast<char*>(z_bytes.data()), 32);

  const uint8_t KYBER_K = 2; // ML-KEM-512 uses k=2
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Read expected keys
  std::vector<uint8_t> expected_ek(ek_size);
  std::vector<uint8_t> expected_dk(dk_size);
  ek_file.read(reinterpret_cast<char*>(expected_ek.data()), ek_size);
  dk_file.read(reinterpret_cast<char*>(expected_dk.data()), dk_size);

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_d, 64);
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  // Copy d and z to device
  cudaMemcpy(d_d, d_bytes.data(), 32, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d + 32, z_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Launch kernel
  ml_kem_keygen_kernel<KYBER_K, 3><<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(ek_size);
  std::vector<uint8_t> h_dk(dk_size);
  cudaMemcpy(h_ek.data(), d_ek, ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, dk_size, cudaMemcpyDeviceToHost);

  // Compare results
  ASSERT_EQ(h_ek, expected_ek) << "Encapsulation key mismatch for ML-KEM-512";
  ASSERT_EQ(h_dk, expected_dk) << "Decapsulation key mismatch for ML-KEM-512";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen768)
{
  // Read test data files
  std::ifstream random_file(TEST_DATA_PATH + "/ml_kem_768_data/ml_kem_768_8192_random_bytes.txt", std::ios::binary);
  std::ifstream ek_file(TEST_DATA_PATH + "/ml_kem_768_data/ml_kem_768_8192_ek.txt", std::ios::binary);
  std::ifstream dk_file(TEST_DATA_PATH + "/ml_kem_768_data/ml_kem_768_8192_dk.txt", std::ios::binary);

  ASSERT_TRUE(random_file.is_open()) << "Failed to open random bytes file";
  ASSERT_TRUE(ek_file.is_open()) << "Failed to open encryption key file";
  ASSERT_TRUE(dk_file.is_open()) << "Failed to open decryption key file";

  // Read first key pair data
  std::vector<uint8_t> d_bytes(32);
  std::vector<uint8_t> z_bytes(32);
  random_file.read(reinterpret_cast<char*>(d_bytes.data()), 32);
  random_file.read(reinterpret_cast<char*>(z_bytes.data()), 32);

  const uint8_t KYBER_K = 3; // ML-KEM-768 uses k=3
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Read expected keys
  std::vector<uint8_t> expected_ek(ek_size);
  std::vector<uint8_t> expected_dk(dk_size);
  ek_file.read(reinterpret_cast<char*>(expected_ek.data()), ek_size);
  dk_file.read(reinterpret_cast<char*>(expected_dk.data()), dk_size);

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_d, 64);
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  // Copy d and z to device
  cudaMemcpy(d_d, d_bytes.data(), 32, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d + 32, z_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Launch kernel
  ml_kem_keygen_kernel<KYBER_K, 2><<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(ek_size);
  std::vector<uint8_t> h_dk(dk_size);
  cudaMemcpy(h_ek.data(), d_ek, ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, dk_size, cudaMemcpyDeviceToHost);

  // Compare results
  ASSERT_EQ(h_ek, expected_ek) << "Encapsulation key mismatch for ML-KEM-768";
  ASSERT_EQ(h_dk, expected_dk) << "Decapsulation key mismatch for ML-KEM-768";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen1024)
{
  // Read test data files
  std::ifstream random_file(TEST_DATA_PATH + "/ml_kem_1024_data/ml_kem_1024_8192_random_bytes.txt", std::ios::binary);
  std::ifstream ek_file(TEST_DATA_PATH + "/ml_kem_1024_data/ml_kem_1024_8192_ek.txt", std::ios::binary);
  std::ifstream dk_file(TEST_DATA_PATH + "/ml_kem_1024_data/ml_kem_1024_8192_dk.txt", std::ios::binary);

  ASSERT_TRUE(random_file.is_open()) << "Failed to open random bytes file";
  ASSERT_TRUE(ek_file.is_open()) << "Failed to open encryption key file";
  ASSERT_TRUE(dk_file.is_open()) << "Failed to open decryption key file";

  // Read first key pair data
  std::vector<uint8_t> d_bytes(32);
  std::vector<uint8_t> z_bytes(32);
  random_file.read(reinterpret_cast<char*>(d_bytes.data()), 32);
  random_file.read(reinterpret_cast<char*>(z_bytes.data()), 32);

  const uint8_t KYBER_K = 4; // ML-KEM-1024 uses k=4
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Read expected keys
  std::vector<uint8_t> expected_ek(ek_size);
  std::vector<uint8_t> expected_dk(dk_size);
  ek_file.read(reinterpret_cast<char*>(expected_ek.data()), ek_size);
  dk_file.read(reinterpret_cast<char*>(expected_dk.data()), dk_size);

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_d, 64);
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  // Copy d and z to device
  cudaMemcpy(d_d, d_bytes.data(), 32, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d + 32, z_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Launch kernel
  ml_kem_keygen_kernel<KYBER_K, 2><<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(ek_size);
  std::vector<uint8_t> h_dk(dk_size);
  cudaMemcpy(h_ek.data(), d_ek, ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, dk_size, cudaMemcpyDeviceToHost);

  // Compare results
  ASSERT_EQ(h_ek, expected_ek) << "Encapsulation key mismatch for ML-KEM-1024";
  ASSERT_EQ(h_dk, expected_dk) << "Decapsulation key mismatch for ML-KEM-1024";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen512_Batch)
{
  const uint8_t KYBER_K = 2; // ML-KEM-512 uses k=2
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Read test data files
  std::ifstream random_file(TEST_DATA_PATH + "/ml_kem_512_data/ml_kem_512_8192_random_bytes.txt", std::ios::binary);
  std::ifstream ek_file(TEST_DATA_PATH + "/ml_kem_512_data/ml_kem_512_8192_ek.txt", std::ios::binary);
  std::ifstream dk_file(TEST_DATA_PATH + "/ml_kem_512_data/ml_kem_512_8192_dk.txt", std::ios::binary);

  ASSERT_TRUE(random_file.is_open()) << "Failed to open random bytes file";
  ASSERT_TRUE(ek_file.is_open()) << "Failed to open encryption key file";
  ASSERT_TRUE(dk_file.is_open()) << "Failed to open decryption key file";

  // Read all random bytes
  std::vector<uint8_t> random_bytes(BATCH_SIZE * 64); // 32 bytes d + 32 bytes z for each key
  random_file.read(reinterpret_cast<char*>(random_bytes.data()), BATCH_SIZE * 64);

  // Read all expected keys
  std::vector<uint8_t> expected_eks(BATCH_SIZE * ek_size);
  std::vector<uint8_t> expected_dks(BATCH_SIZE * dk_size);
  ek_file.read(reinterpret_cast<char*>(expected_eks.data()), BATCH_SIZE * ek_size);
  dk_file.read(reinterpret_cast<char*>(expected_dks.data()), BATCH_SIZE * dk_size);

  // Allocate device memory
  uint8_t* d_random;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_random, BATCH_SIZE * 64);
  cudaMalloc((void**)&d_ek, BATCH_SIZE * ek_size);
  cudaMalloc((void**)&d_dk, BATCH_SIZE * dk_size);
  cudaMalloc((void**)&d_A, BATCH_SIZE * A_size * sizeof(Zq));

  // Copy random bytes to device
  cudaMemcpy(d_random, random_bytes.data(), BATCH_SIZE * 64, cudaMemcpyHostToDevice);

  // Launch kernel with one block per key
  ml_kem_keygen_kernel<KYBER_K, 3><<<BATCH_SIZE, 128>>>(d_random, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(BATCH_SIZE * ek_size);
  std::vector<uint8_t> h_dk(BATCH_SIZE * dk_size);
  cudaMemcpy(h_ek.data(), d_ek, BATCH_SIZE * ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, BATCH_SIZE * dk_size, cudaMemcpyDeviceToHost);

  // Compare results
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    std::vector<uint8_t> current_ek(h_ek.begin() + i * ek_size, h_ek.begin() + (i + 1) * ek_size);
    std::vector<uint8_t> current_dk(h_dk.begin() + i * dk_size, h_dk.begin() + (i + 1) * dk_size);
    std::vector<uint8_t> expected_ek(expected_eks.begin() + i * ek_size, expected_eks.begin() + (i + 1) * ek_size);
    std::vector<uint8_t> expected_dk(expected_dks.begin() + i * dk_size, expected_dks.begin() + (i + 1) * dk_size);

    if (true) { // Debug output toggle
      if (current_ek != expected_ek) {
        std::cout << "First EK mismatch at index " << i << "\nExpected EK: ";
        for (const auto& byte : expected_ek) {
          printf("%02x", byte);
        }
        std::cout << "\nActual EK: ";
        for (const auto& byte : current_ek) {
          printf("%02x", byte);
        }
        std::cout << std::endl;
      }
      if (current_dk != expected_dk) {
        std::cout << "First DK mismatch at index " << i << "\nExpected DK: ";
        for (const auto& byte : expected_dk) {
          printf("%02x", byte);
        }
        std::cout << "\nActual DK: ";
        for (const auto& byte : current_dk) {
          printf("%02x", byte);
        }
        std::cout << std::endl;
      }
    }

    ASSERT_EQ(current_ek, expected_ek) << "Encapsulation key mismatch for ML-KEM-512 at index " << i;
    ASSERT_EQ(current_dk, expected_dk) << "Decapsulation key mismatch for ML-KEM-512 at index " << i;
  }

  cudaFree(d_random);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen768_Batch)
{
  const uint8_t KYBER_K = 3; // ML-KEM-768 uses k=3
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Read test data files
  std::ifstream random_file(TEST_DATA_PATH + "/ml_kem_768_data/ml_kem_768_8192_random_bytes.txt", std::ios::binary);
  std::ifstream ek_file(TEST_DATA_PATH + "/ml_kem_768_data/ml_kem_768_8192_ek.txt", std::ios::binary);
  std::ifstream dk_file(TEST_DATA_PATH + "/ml_kem_768_data/ml_kem_768_8192_dk.txt", std::ios::binary);

  ASSERT_TRUE(random_file.is_open()) << "Failed to open random bytes file";
  ASSERT_TRUE(ek_file.is_open()) << "Failed to open encryption key file";
  ASSERT_TRUE(dk_file.is_open()) << "Failed to open decryption key file";

  // Read all random bytes
  std::vector<uint8_t> random_bytes(BATCH_SIZE * 64); // 32 bytes d + 32 bytes z for each key
  random_file.read(reinterpret_cast<char*>(random_bytes.data()), BATCH_SIZE * 64);

  // Read all expected keys
  std::vector<uint8_t> expected_eks(BATCH_SIZE * ek_size);
  std::vector<uint8_t> expected_dks(BATCH_SIZE * dk_size);
  ek_file.read(reinterpret_cast<char*>(expected_eks.data()), BATCH_SIZE * ek_size);
  dk_file.read(reinterpret_cast<char*>(expected_dks.data()), BATCH_SIZE * dk_size);

  // Allocate device memory
  uint8_t* d_random;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_random, BATCH_SIZE * 64);
  cudaMalloc((void**)&d_ek, BATCH_SIZE * ek_size);
  cudaMalloc((void**)&d_dk, BATCH_SIZE * dk_size);
  cudaMalloc((void**)&d_A, BATCH_SIZE * A_size * sizeof(Zq));

  // Copy random bytes to device
  cudaMemcpy(d_random, random_bytes.data(), BATCH_SIZE * 64, cudaMemcpyHostToDevice);

  // Launch kernel with one block per key
  ml_kem_keygen_kernel<KYBER_K, 2><<<BATCH_SIZE, 128>>>(d_random, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(BATCH_SIZE * ek_size);
  std::vector<uint8_t> h_dk(BATCH_SIZE * dk_size);
  cudaMemcpy(h_ek.data(), d_ek, BATCH_SIZE * ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, BATCH_SIZE * dk_size, cudaMemcpyDeviceToHost);

  // Compare results
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    std::vector<uint8_t> current_ek(h_ek.begin() + i * ek_size, h_ek.begin() + (i + 1) * ek_size);
    std::vector<uint8_t> current_dk(h_dk.begin() + i * dk_size, h_dk.begin() + (i + 1) * dk_size);
    std::vector<uint8_t> expected_ek(expected_eks.begin() + i * ek_size, expected_eks.begin() + (i + 1) * ek_size);
    std::vector<uint8_t> expected_dk(expected_dks.begin() + i * dk_size, expected_dks.begin() + (i + 1) * dk_size);

    ASSERT_EQ(current_ek, expected_ek) << "Encapsulation key mismatch for ML-KEM-768 at index " << i;
    ASSERT_EQ(current_dk, expected_dk) << "Decapsulation key mismatch for ML-KEM-768 at index " << i;
  }

  cudaFree(d_random);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen1024_Batch)
{
  const uint8_t KYBER_K = 4; // ML-KEM-1024 uses k=4
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Read test data files
  std::ifstream random_file(TEST_DATA_PATH + "/ml_kem_1024_data/ml_kem_1024_8192_random_bytes.txt", std::ios::binary);
  std::ifstream ek_file(TEST_DATA_PATH + "/ml_kem_1024_data/ml_kem_1024_8192_ek.txt", std::ios::binary);
  std::ifstream dk_file(TEST_DATA_PATH + "/ml_kem_1024_data/ml_kem_1024_8192_dk.txt", std::ios::binary);

  ASSERT_TRUE(random_file.is_open()) << "Failed to open random bytes file";
  ASSERT_TRUE(ek_file.is_open()) << "Failed to open encryption key file";
  ASSERT_TRUE(dk_file.is_open()) << "Failed to open decryption key file";

  // Read all random bytes
  std::vector<uint8_t> random_bytes(BATCH_SIZE * 64); // 32 bytes d + 32 bytes z for each key
  random_file.read(reinterpret_cast<char*>(random_bytes.data()), BATCH_SIZE * 64);

  // Read all expected keys
  std::vector<uint8_t> expected_eks(BATCH_SIZE * ek_size);
  std::vector<uint8_t> expected_dks(BATCH_SIZE * dk_size);
  ek_file.read(reinterpret_cast<char*>(expected_eks.data()), BATCH_SIZE * ek_size);
  dk_file.read(reinterpret_cast<char*>(expected_dks.data()), BATCH_SIZE * dk_size);

  // Allocate device memory
  uint8_t* d_random;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_random, BATCH_SIZE * 64);
  cudaMalloc((void**)&d_ek, BATCH_SIZE * ek_size);
  cudaMalloc((void**)&d_dk, BATCH_SIZE * dk_size);
  cudaMalloc((void**)&d_A, BATCH_SIZE * A_size * sizeof(Zq));

  // Copy random bytes to device
  cudaMemcpy(d_random, random_bytes.data(), BATCH_SIZE * 64, cudaMemcpyHostToDevice);

  // Launch kernel with one block per key
  ml_kem_keygen_kernel<KYBER_K, 2><<<BATCH_SIZE, 128>>>(d_random, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(BATCH_SIZE * ek_size);
  std::vector<uint8_t> h_dk(BATCH_SIZE * dk_size);
  cudaMemcpy(h_ek.data(), d_ek, BATCH_SIZE * ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, BATCH_SIZE * dk_size, cudaMemcpyDeviceToHost);

  // Compare results
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    std::vector<uint8_t> current_ek(h_ek.begin() + i * ek_size, h_ek.begin() + (i + 1) * ek_size);
    std::vector<uint8_t> current_dk(h_dk.begin() + i * dk_size, h_dk.begin() + (i + 1) * dk_size);
    std::vector<uint8_t> expected_ek(expected_eks.begin() + i * ek_size, expected_eks.begin() + (i + 1) * ek_size);
    std::vector<uint8_t> expected_dk(expected_dks.begin() + i * dk_size, expected_dks.begin() + (i + 1) * dk_size);

    ASSERT_EQ(current_ek, expected_ek) << "Encapsulation key mismatch for ML-KEM-1024 at index " << i;
    ASSERT_EQ(current_dk, expected_dk) << "Decapsulation key mismatch for ML-KEM-1024 at index " << i;
  }

  cudaFree(d_random);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(MLKemKeygenTest, ML_KEM_Internal_Keygen1024_Benchmark)
{
  const uint8_t KYBER_K = 3; // ML-KEM-1024 uses k=4
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  const unsigned int batch_size = 1 << 20;

  // Read all random bytes
  std::vector<uint8_t> random_bytes(batch_size * 64); // 32 bytes d + 32 bytes z for each key

  // Allocate device memory
  uint8_t* d_random;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;

  cudaMalloc((void**)&d_random, batch_size * 64);
  cudaMalloc((void**)&d_ek, batch_size * ek_size);
  cudaMalloc((void**)&d_dk, batch_size * dk_size);
  cudaMalloc((void**)&d_A, batch_size * A_size * sizeof(Zq));

  // Copy random bytes to device
  cudaMemcpy(d_random, random_bytes.data(), batch_size * 64, cudaMemcpyHostToDevice);

  // Launch kernel with one block per key
  ml_kem_keygen_kernel<KYBER_K, 2><<<batch_size, 128>>>(d_random, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  cudaFree(d_random);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}