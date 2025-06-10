#include <cuda_runtime.h>
#include "ml_kem/ring/cuda_zq.cuh"

#include "ml_kem/cuda_ml_kem_kernels.cuh"

#include <gtest/gtest.h>
#include <iostream>
#include <random>

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

class MLKemTest : public IcicleTestBase
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

  template <const uint8_t k, const uint8_t eta1, const uint8_t eta2, const uint8_t du, const uint8_t dv>
  void key_check_test()
  {
    constexpr size_t ek_size = 384 * k + 32;
    constexpr size_t dk_size = 768 * k + 96;

    uint8_t d_buffer[64];
    uint8_t m_buffer[32];
    uint8_t* d_d;
    uint8_t* d_z;
    uint8_t* d_m;
    cudaMalloc(&d_d, sizeof(d_buffer) * BATCH_SIZE);
    cudaMalloc(&d_m, sizeof(m_buffer) * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
      randomize(d_buffer, 64);
      randomize(m_buffer, 32);
      cudaMemcpy(d_d + i * 64, d_buffer, 64, cudaMemcpyHostToDevice);
      cudaMemcpy(d_m + i * 32, m_buffer, 32, cudaMemcpyHostToDevice);
    }
    uint8_t* d_ek;
    uint8_t* d_dk;
    Zq* d_A;
    cudaMalloc(&d_ek, ek_size * BATCH_SIZE);
    cudaMalloc(&d_dk, dk_size * BATCH_SIZE);
    cudaMalloc(&d_A, PolyMatrix<256, k, k, Zq>::byte_size() * BATCH_SIZE);

    ml_kem_keygen_kernel<k, eta1><<<BATCH_SIZE, 128>>>(d_d, d_ek, d_dk, d_A);

    uint8_t* d_K;
    uint8_t* d_c;
    cudaMalloc(&d_K, 32 * BATCH_SIZE);
    cudaMalloc(&d_c, 32 * (du * k + dv) * BATCH_SIZE);

    ml_kem_encaps_kernel<k, eta1, eta2, du, dv><<<BATCH_SIZE, 128>>>(d_ek, d_m, d_K, d_c, d_A);

    uint8_t* d_K_prime;
    cudaMalloc(&d_K_prime, 32 * BATCH_SIZE);
    ml_kem_decaps_kernel<k, eta1, eta2, du, dv><<<BATCH_SIZE, 128>>>(d_dk, d_c, d_K_prime, d_A);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("Kernel launch failed with error \"%s\"", cudaGetErrorString(err)); }

    uint8_t h_K[32];
    uint8_t h_K_prime[32];
    for (int i = 0; i < BATCH_SIZE; ++i) {
      cudaMemcpy(h_K, d_K + i * 32, 32, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_K_prime, d_K_prime + i * 32, 32, cudaMemcpyDeviceToHost);
      for (int j = 0; j < 32; ++j) {
        ASSERT_EQ(h_K[j], h_K_prime[j]) << "K check mismatch: i = " << i << " j = " << j;
      }
    }

    cudaFree(d_d);
    cudaFree(d_z);
    cudaFree(d_m);
    cudaFree(d_ek);
    cudaFree(d_dk);
    cudaFree(d_A);
    cudaFree(d_K);
    cudaFree(d_c);
    cudaFree(d_K_prime);
  }
};

const size_t MLKemTest::BATCH_SIZE = 8192;

TEST_F(MLKemTest, KeyCheckTest512Batch)
{
  constexpr uint8_t du = 10;
  constexpr uint8_t dv = 4;
  constexpr uint8_t k = 2;
  constexpr uint8_t eta1 = 3;
  constexpr uint8_t eta2 = 2;
  key_check_test<k, eta1, eta2, du, dv>();
}

TEST_F(MLKemTest, KeyCheckTest768Batch)
{
  constexpr uint8_t du = 10;
  constexpr uint8_t dv = 4;
  constexpr uint8_t k = 3;
  constexpr uint8_t eta1 = 2;
  constexpr uint8_t eta2 = 2;
  key_check_test<k, eta1, eta2, du, dv>();
}

TEST_F(MLKemTest, KeyCheckTest1024Batch)
{
  constexpr uint8_t du = 11;
  constexpr uint8_t dv = 5;
  constexpr uint8_t k = 4;
  constexpr uint8_t eta1 = 2;
  constexpr uint8_t eta2 = 2;
  key_check_test<k, eta1, eta2, du, dv>();
}