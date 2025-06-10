#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/cuda_ml_kem_kernels.cuh"

#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/hash/keccak.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "test_base.h"
#include "icicle/utils/rand_gen.h"

using namespace icicle;
using namespace icicle::pqc::ml_kem;

class KyberTest : public IcicleTestBase
{
public:
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
      // Use fully qualified names for std::hex, std::setw, and std::setfill
      hexStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byteData[i]);
    }

    return hexStream.str();
  }
};

// __global__
// void theta_test_kernel1(const uint64_t* state, uint64_t* out){
//   __shared__ uint64_t s[KECCAK_STATE_SIZE];
//   for (int i = 0; i < 25; i++){ s[i] = state[i]; }

//   uint64_t t0, t1, t2, t3, t4, t5;

//   __syncthreads();
//   THETA(
//     s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
//     s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);

//   __syncthreads();
//   for (int i = 0; i < 25; i++){ out[i] = s[i]; }
// }

// __global__
// void theta_test_kernel2(const uint64_t* state, uint64_t* out){
//   uint64_t s = threadIdx.x < 25 ? state[threadIdx.x] : 0;

//   __syncthreads();
//   uint64_t t = theta(s);
//   __syncthreads();

//   if (threadIdx.x < 25){ out[threadIdx.x] = t; }
// }

// TEST_F(KyberTest, Sha3Theta)
// {
//   uint64_t state[KECCAK_STATE_SIZE];
//   for (int i = 0; i < KECCAK_STATE_SIZE; i++){ state[i] = i; }

//   uint64_t* d_state;
//   cudaMalloc((void**)&d_state, KECCAK_STATE_SIZE * sizeof(uint64_t));
//   cudaMemcpy(d_state, state, KECCAK_STATE_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

//   size_t output_size = KECCAK_STATE_SIZE * sizeof(uint64_t);

//   uint64_t* d_out1;
//   cudaMalloc((void**)&d_out1, output_size);

//   uint64_t* d_out2;
//   cudaMalloc((void**)&d_out2, output_size);

//   theta_test_kernel1<<<1,1>>>(d_state, d_out1);
//   theta_test_kernel2<<<1,32>>>(d_state, d_out2);

//   uint64_t out1[KECCAK_STATE_SIZE];
//   cudaMemcpy(&out1, d_out1, output_size, cudaMemcpyDeviceToHost);

//   uint64_t out2[KECCAK_STATE_SIZE];
//   cudaMemcpy(&out2, d_out2, output_size, cudaMemcpyDeviceToHost);

//   ASSERT_EQ(0, memcmp(out1, out2, output_size));
//   // for (int i = 0; i < KECCAK_STATE_SIZE; i++)
//   // {
//   //   if (out1[i] != out2[i])
//   //   {
//   //     printf("%d: %lu != %lu\n", i, out1[i], out2[i]);
//   //   }
//   // }

//   cudaFree(d_state);
//   cudaFree(d_out1);
//   cudaFree(d_out2);
// }

__global__ void absorb_test_kernel(const uint8_t d[32], uint64_t* output)
{
  if (threadIdx.x >= 25) { return; }

  // Test absorb with SHA3-512 configuration and extra_input=7
  uint64_t s = absorb<32, SHA3_512_RATE, (((uint32_t)SHA3_DELIM_BITS) << 8), SHA3_DELIM_SUFFIX>(d, 7);
  output[threadIdx.x] = s;
}

TEST_F(KyberTest, Sha3Absorb)
{
  // Create test input
  uint8_t input[32];
  for (int i = 0; i < 32; i++) {
    input[i] = i;
  }

  // Allocate device memory
  uint8_t* d_input;
  cudaMalloc((void**)&d_input, 32);
  cudaMemcpy(d_input, input, 32, cudaMemcpyHostToDevice);

  uint64_t* d_output;
  cudaMalloc((void**)&d_output, 25 * sizeof(uint64_t));

  // Run kernel
  absorb_test_kernel<<<1, 32>>>(d_input, d_output);

  // Copy results back
  uint64_t output[25];
  cudaMemcpy(output, d_output, 25 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // Verify first block contains input data
  for (int i = 0; i < 4; i++) {
    uint64_t expected = *reinterpret_cast<uint64_t*>(&input[i * 8]);
    ASSERT_EQ(output[i], expected);
  }

  // Verify padding with extra_input=7
  ASSERT_EQ(output[4], (uint64_t)0x0607);

  // Verify last byte has suffix
  ASSERT_EQ(output[8] & 0x8000000000000000ULL, 0x8000000000000000ULL);

  // Verify remaining state is zero
  for (int i = 9; i < 25; i++) {
    ASSERT_EQ(output[i], 0);
  }

  cudaFree(d_input);
  cudaFree(d_output);
}

__global__ void sha3_512_test_kernel(const uint8_t d[32], uint64_t* output)
{
  // Initialize state with absorbed input and padding
  uint64_t s = absorb<32, SHA3_512_RATE, ((uint32_t)SHA3_DELIM_BITS), SHA3_DELIM_SUFFIX>(d, 0);
  __syncthreads();

  // Run keccakf permutation
  uint64_t t = keccakf(s);

  // Save result
  if (threadIdx.x < 25) { output[threadIdx.x] = t; }
}

template <uint8_t k>
__global__ void keygen_G_test_kernel(const uint8_t d[32], uint64_t* output)
{
  __shared__ __align__(8) uint64_t rho_sigma[8]; // rho = rho_sigma[0 : 4], sigma = rho_sigma[4 : 8]
  G<k>(d, rho_sigma);
  __syncthreads();
  if (threadIdx.x < 8) { output[threadIdx.x] = rho_sigma[threadIdx.x]; }
}

TEST_F(KyberTest, G)
{
  const std::string d = "1af17a664e3fa8e419b8ba05c2a173169df76162a5a286e0c405b460d478f7ef";
  const std::string expected_rho = "ec6e912ca1729d2ec0c233409baa3d6b105eb7f862e97ad86a367a8b798c325c";
  const std::string expected_sigma = "1a3b046be3aec19ef12d23ab6afbb2c2319f6764b35ca793422288211ad2b708";

  // Convert hex strings to byte arrays
  uint8_t d_bytes[32];
  for (int i = 0; i < 32; i++) {
    d_bytes[i] = std::stoi(d.substr(i * 2, 2), nullptr, 16);
  }

  // Allocate device memory
  uint8_t* d_d;
  uint64_t* d_output;
  cudaMalloc(&d_d, 32);
  cudaMalloc(&d_output, 8 * sizeof(uint64_t));

  // Copy input to device
  cudaMemcpy(d_d, d_bytes, 32, cudaMemcpyHostToDevice);

  // Launch kernel
  keygen_G_test_kernel<2><<<1, 32>>>(d_d, d_output);

  // Copy results back
  uint64_t output[8];
  cudaMemcpy(output, d_output, 8 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // Convert output bytes to hex strings
  std::stringstream ss;
  uint8_t* output_bytes = (uint8_t*)output;

  for (int i = 0; i < 64; i++) {
    ss << std::hex << std::setfill('0') << std::setw(2) << (int)output_bytes[i];
  }

  std::string result = ss.str();
  std::string actual_rho = result.substr(0, 64);
  std::string actual_sigma = result.substr(64, 64);

  bool rho_matches = (actual_rho == expected_rho);
  bool sigma_matches = (actual_sigma == expected_sigma);

  if (!rho_matches) {
    std::cout << "Rho mismatch:\n";
    std::cout << "Expected: " << expected_rho << "\n";
    std::cout << "Got:      " << actual_rho << "\n";
  }

  if (!sigma_matches) {
    std::cout << "Sigma mismatch:\n";
    std::cout << "Expected: " << expected_sigma << "\n";
    std::cout << "Got:      " << actual_sigma << "\n";
  }

  ASSERT_TRUE(rho_matches) << "Rho value does not match expected";
  ASSERT_TRUE(sigma_matches) << "Sigma value does not match expected";

  // Cleanup
  cudaFree(d_d);
  cudaFree(d_output);
}

__global__ void shake128_test_kernel(const uint8_t* d, uint8_t* output, uint32_t extra_input)
{
  // First thread loads input into state
  uint64_t s = absorb<32, SHAKE_128_RATE, (SHAKE_DELIM_BITS << 16), SHAKE_DELIM_SUFFIX>(d, extra_input);

  __syncthreads();

  // Run Keccak-f
  s = keccakf(s);

  if (threadIdx.x < 8) { ((uint64_t*)output)[threadIdx.x] = s; }
}

TEST_F(KyberTest, Shake128)
{
  const std::string input = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const uint32_t extra_input = 0x0101;
  const std::string expected = "df292bc1b69e64171d8817539d31ce0e68be644d837f10cf37560424136586600e1c2d2327d36fa053dd148"
                               "8f9109c07631747c9c6dd04d5330ff5533e971fac";

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_output;
  cudaMalloc(&d_d, 32);
  cudaMalloc(&d_output, 64);

  // Convert hex string to bytes
  std::vector<uint8_t> input_bytes(32);
  for (int i = 0; i < 32; i++) {
    input_bytes[i] = std::stoi(input.substr(i * 2, 2), nullptr, 16);
  }

  // Copy input to device
  cudaMemcpy(d_d, input_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Run kernel
  shake128_test_kernel<<<1, 32>>>(d_d, d_output, extra_input);

  // Copy results back
  std::vector<uint8_t> output(64);
  cudaMemcpy(output.data(), d_output, 64, cudaMemcpyDeviceToHost);

  // Convert output to hex string
  std::stringstream ss;
  for (int i = 0; i < 64; i++) {
    ss << std::hex << std::setfill('0') << std::setw(2) << (int)output[i];
  }

  ASSERT_EQ(expected, ss.str());

  // Cleanup
  cudaFree(d_d);
  cudaFree(d_output);
}

const int16_t reference_zetas[128] = {
  1,    1729, 2580, 3289, 2642, 630,  1897, 848,  1062, 1919, 193,  797,  2786, 3260, 569,  1746, 296,  2447, 1339,
  1476, 3046, 56,   2240, 1333, 1426, 2094, 535,  2882, 2393, 2879, 1974, 821,  289,  331,  3253, 1756, 1197, 2304,
  2277, 2055, 650,  1977, 2513, 632,  2865, 33,   1320, 1915, 2319, 1435, 807,  452,  1438, 2868, 1534, 2402, 2647,
  2617, 1481, 648,  2474, 3110, 1227, 910,  17,   2761, 583,  2649, 1637, 723,  2288, 1100, 1409, 2662, 3281, 233,
  756,  2156, 3015, 3050, 1703, 1651, 2789, 1789, 1847, 952,  1461, 2687, 939,  2308, 2437, 2388, 733,  2337, 268,
  641,  1584, 2298, 2037, 3220, 375,  2549, 2090, 1645, 1063, 319,  2773, 757,  2099, 561,  2466, 2594, 2804, 1092,
  403,  1026, 1143, 2150, 2775, 886,  1722, 1212, 1874, 1029, 2110, 2935, 885,  2154};

int16_t reference_montgomery_reduce(int32_t a)
{
  int16_t t;

  t = a % 3329;
  return t;
}

/*************************************************
 * Name:        fqmul
 *
 * Description: Multiplication followed by Montgomery reduction
 *
 * Arguments:   - int16_t a: first factor
 *              - int16_t b: second factor
 *
 * Returns 16-bit integer congruent to a*b*R^{-1} mod q
 **************************************************/
static int16_t reference_fqmul(int16_t a, int16_t b) { return reference_montgomery_reduce((int32_t)a * b); }

/*************************************************
 * Name:        ntt
 *
 * Description: Inplace number-theoretic transform (NTT) in Rq.
 *              input is in standard order, output is in bitreversed order
 *
 * Arguments:   - int16_t r[256]: pointer to input/output vector of elements of Zq
 **************************************************/
void reference_ntt(int16_t r[256])
{
  unsigned int len, start, j, k;
  int16_t t, zeta;
  constexpr uint16_t Q = 3329;

  k = 1;
  for (len = 128; len >= 2; len >>= 1) {
    for (start = 0; start < 256; start = j + len) {
      zeta = reference_zetas[k++];
      for (j = start; j < start + len; j++) {
        t = (int32_t)zeta * r[j + len] % Q;
        r[j + len] = (r[j] + Q - t) % Q;
        r[j] = (r[j] + t) % Q;
      }
    }
  }
}

void reference_basemul(int16_t r[2], const int16_t a[2], const int16_t b[2], int16_t zeta)
{
  r[0] = reference_fqmul(a[1], b[1]);
  r[0] = reference_fqmul(r[0], zeta);
  r[0] += reference_fqmul(a[0], b[0]);
  r[1] = reference_fqmul(a[0], b[1]);
  r[1] += reference_fqmul(a[1], b[0]);
}

typedef struct {
  int16_t coeffs[256];
} reference_poly;

void reference_poly_basemul_montgomery(reference_poly* r, const reference_poly* a, const reference_poly* b)
{
  unsigned int i;
  for (i = 0; i < 256 / 4; i++) {
    reference_basemul(&r->coeffs[4 * i], &a->coeffs[4 * i], &b->coeffs[4 * i], reference_zetas[64 + i]);
    reference_basemul(
      &r->coeffs[4 * i + 2], &a->coeffs[4 * i + 2], &b->coeffs[4 * i + 2], 3329 - reference_zetas[64 + i]);
  }
}

template <uint8_t k>
__launch_bounds__(128) __global__ void nttInplaceKernel(Zq* array)
{
  __shared__ Zq shared_array[256];
  for (int i = 0; i < k; ++i) {
    shared_array[threadIdx.x * 2] = array[threadIdx.x * 2 + i * 256];
    shared_array[threadIdx.x * 2 + 1] = array[threadIdx.x * 2 + 1 + i * 256];
    __syncthreads();
    Poly<256, Zq> poly(shared_array);
    ntt_inplace(poly);
    __syncthreads();
    array[threadIdx.x * 2 + i * 256] = shared_array[threadIdx.x * 2];
    array[threadIdx.x * 2 + 1 + i * 256] = shared_array[threadIdx.x * 2 + 1];
  }
}

template <uint8_t k>
__launch_bounds__(128) __global__ void inttInplaceKernel(Zq* array)
{
  __shared__ Zq shared_array[256];
  for (int i = 0; i < k; ++i) {
    shared_array[threadIdx.x * 2] = array[threadIdx.x * 2 + i * 256];
    shared_array[threadIdx.x * 2 + 1] = array[threadIdx.x * 2 + 1 + i * 256];
    __syncthreads();
    Poly<256, Zq> poly(shared_array);
    intt_inplace(poly);
    __syncthreads();
    array[threadIdx.x * 2 + i * 256] = shared_array[threadIdx.x * 2];
    array[threadIdx.x * 2 + 1 + i * 256] = shared_array[threadIdx.x * 2 + 1];
  }
}

TEST_F(KyberTest, INTTReferenceAgainstTestVectors)
{
  const int16_t reference_u_ntt[512] = {
    0x10c, 0x6c9, 0xabb, 0x615, 0xff,  0xb09, 0x5af, 0x10f, 0xa8a, 0xcb5, 0xad0, 0xc78, 0x396, 0xb6a, 0x8bb, 0x1f1,
    0x53d, 0x576, 0x36f, 0x64a, 0x79d, 0x6c8, 0x3f9, 0xe6,  0x7f9, 0x586, 0xce5, 0x437, 0x792, 0x52d, 0x6f9, 0x7df,
    0x9e7, 0x6b8, 0xc8a, 0x4eb, 0x47c, 0x924, 0x85,  0x2f,  0x984, 0xcae, 0xbf,  0x871, 0x28c, 0x6b4, 0x2bc, 0x202,
    0x1c4, 0xc43, 0x99,  0x33f, 0x742, 0x24e, 0x236, 0x2c,  0x4a2, 0x25e, 0x37e, 0x8d0, 0x7f2, 0x7b6, 0xcb0, 0x309,
    0xabc, 0x2d5, 0xb70, 0x4ad, 0x857, 0xa56, 0x4d1, 0x550, 0xa02, 0x581, 0xaf2, 0x1e5, 0x5ed, 0x29f, 0x69b, 0xce6,
    0x20a, 0x574, 0xa17, 0xca3, 0xc4d, 0x1f1, 0xc06, 0x595, 0x488, 0x8e3, 0xcfc, 0x44b, 0x444, 0xbf4, 0xad8, 0x20f,
    0x46e, 0xa62, 0x16b, 0xad6, 0x191, 0x7c1, 0x835, 0x364, 0x994, 0x9df, 0x5d6, 0xc61, 0xcd0, 0x86d, 0x508, 0xc6e,
    0x93b, 0x8fe, 0x972, 0x491, 0x4db, 0x853, 0x948, 0x768, 0x970, 0xb38, 0x8d0, 0xd9,  0x656, 0x809, 0xa07, 0x553,
    0x689, 0x24a, 0x908, 0x850, 0x71e, 0x133, 0x30f, 0xa6f, 0x28e, 0x252, 0xce4, 0xb2c, 0x9db, 0xfd,  0x685, 0x57d,
    0x754, 0x8b4, 0x894, 0xaf4, 0x208, 0x8dd, 0xc6b, 0xe2,  0xa05, 0x6c2, 0xaac, 0x3ec, 0x8b,  0x4c,  0x15a, 0xcb4,
    0x421, 0x633, 0x75e, 0xc5f, 0x9d9, 0xb25, 0xb01, 0x2f6, 0x15,  0x8f2, 0x80b, 0xb9c, 0x571, 0x27f, 0xb92, 0x85b,
    0xc3e, 0x3b8, 0x46c, 0x911, 0x701, 0xa0e, 0x2af, 0x2c,  0x204, 0xcb,  0x277, 0x94d, 0x8dd, 0xc39, 0x6c5, 0x350,
    0x6b0, 0xba2, 0x7d,  0x9f6, 0x63,  0x205, 0x1f9, 0xb7,  0x8fd, 0x596, 0xc27, 0x539, 0x849, 0x495, 0x5d1, 0x112,
    0x66f, 0xc0a, 0x12f, 0x67c, 0x2c6, 0xa71, 0x688, 0x2e5, 0x4,   0x14b, 0xb31, 0x22c, 0x36,  0x7f5, 0x2fb, 0x673,
    0x25c, 0x539, 0xb56, 0x20,  0x33d, 0xc87, 0xa11, 0x348, 0x95a, 0x965, 0x53c, 0x836, 0xcf,  0xc2,  0x556, 0x380,
    0x25f, 0x9e0, 0x797, 0x31f, 0x6e,  0xac5, 0x745, 0xc9d, 0x67d, 0xad1, 0xc4,  0x156, 0x55f, 0xcc2, 0xc2a, 0xbd1,
    0xe9,  0x673, 0x277, 0x3dd, 0x9be, 0x7c1, 0x669, 0x5c5, 0x832, 0x942, 0xc4a, 0x6c7, 0x763, 0xa1a, 0x4b5, 0x2f2,
    0xa85, 0x664, 0x182, 0x49e, 0x2e6, 0x148, 0x388, 0xbc7, 0xbc,  0x951, 0x1b1, 0x490, 0xb00, 0x89,  0x981, 0x6a1,
    0x6a8, 0xbff, 0x3c0, 0x876, 0x7f2, 0x4e1, 0x3b4, 0x36d, 0x724, 0x423, 0x4f1, 0xed,  0xbef, 0xbe2, 0x8ee, 0x26d,
    0xaa9, 0xafe, 0x2fe, 0x1f2, 0x429, 0x275, 0xb97, 0x171, 0x163, 0xae8, 0x66c, 0x7a5, 0xc03, 0xad5, 0x679, 0xb44,
    0x266, 0x1bc, 0x56d, 0x286, 0x526, 0x1ad, 0x6ac, 0x1ff, 0x344, 0x64f, 0x7fe, 0x7ad, 0x7a,  0x19b, 0xb35, 0x489,
    0xc5b, 0x992, 0x5b8, 0x30c, 0x65f, 0x2d2, 0x52f, 0xcee, 0x884, 0x390, 0xca5, 0x517, 0x16e, 0x14,  0xb17, 0x7ce,
    0x7b2, 0xb90, 0x3a0, 0x13e, 0x7e8, 0x3e6, 0x4e0, 0x3c4, 0x6,   0x13c, 0xb28, 0x77e, 0x366, 0x392, 0x6bc, 0x202,
    0x731, 0x64c, 0x508, 0x1c7, 0xaf7, 0x816, 0xbaf, 0x92,  0xb31, 0x31,  0x110, 0xad6, 0xabf, 0xc83, 0x6b6, 0xc76,
    0xc6b, 0x88c, 0xba9, 0x7b,  0x923, 0x53,  0xa53, 0x4a0, 0x16,  0x60b, 0xba5, 0x433, 0x63d, 0x51f, 0xac3, 0xc50,
    0x44,  0x708, 0x250, 0x8e4, 0xb3f, 0x6f8, 0x852, 0xa65, 0xc2,  0x8bb, 0xc1c, 0xaa9, 0x34d, 0x1c0, 0x60b, 0xc81,
    0x899, 0xcd7, 0xfc,  0x26c, 0x6ac, 0x566, 0x3a,  0x540, 0x8a1, 0x5d4, 0x2a2, 0xcce, 0xcf7, 0x6fb, 0xb2a, 0x647,
    0xc7a, 0x65c, 0x946, 0x435, 0x3d1, 0xa8f, 0x916, 0x99,  0x485, 0x935, 0x46,  0xc49, 0x67,  0x5dd, 0x3ff, 0x99f,
    0xb41, 0xa7f, 0x263, 0x6dc, 0x218, 0x711, 0x8d9, 0x2b9, 0x6a3, 0x53f, 0x58c, 0x897, 0x7dc, 0x978, 0x25c, 0x167,
    0x338, 0x5b6, 0x971, 0x3cd, 0x982, 0x9f9, 0x724, 0x3c,  0x648, 0x62b, 0x8cb, 0xba6, 0x154, 0xc1d, 0x855, 0x975,
    0xa13, 0xb4,  0xae8, 0xa62, 0x31,  0x9fe, 0x572, 0x4e6, 0xc21, 0x8f1, 0x66f, 0x10,  0x39c, 0x3a5, 0xc26, 0xa19,
    0x826, 0xb57, 0x373, 0xb0a, 0x31a, 0x446, 0x681, 0xb09, 0x1ed, 0x7a5, 0x2f8, 0x69f, 0x462, 0x522, 0xbe9, 0x708,
  };
  const int16_t reference_u_intt[512] = {
    0xbbd, 0x350, 0x13,  0x2cb, 0x759, 0x113, 0x1a2, 0xb3a, 0x128, 0x80,  0x3c6, 0x4a7, 0x6a5, 0xfb,  0x87b, 0x285,
    0x57b, 0x42a, 0x29a, 0x92b, 0x60f, 0x2d6, 0x7fd, 0x755, 0x61f, 0x9b4, 0x2b9, 0x56e, 0x290, 0x21f, 0x768, 0x1c2,
    0xa83, 0x4cf, 0x31f, 0xac6, 0x748, 0xcea, 0x64,  0x169, 0xb0f, 0x272, 0x7bc, 0x3e5, 0x8aa, 0x9e5, 0x708, 0xc8a,
    0xa8b, 0x585, 0x38,  0x73e, 0x3e9, 0x26e, 0x57a, 0x5e8, 0x5f6, 0x79d, 0xa3,  0x3d2, 0x80d, 0xc08, 0x227, 0x338,
    0xa38, 0xaad, 0x402, 0x1b7, 0x2c4, 0x3bc, 0x927, 0x2fa, 0xb99, 0x3dc, 0x87b, 0xbcd, 0xc8c, 0xd9,  0xb1e, 0x537,
    0xad4, 0x137, 0x1d1, 0x6a5, 0x408, 0x151, 0x607, 0x847, 0x8be, 0x375, 0x203, 0x168, 0xbfb, 0x826, 0xae5, 0xc16,
    0x18b, 0x8c0, 0xa88, 0x78b, 0x2a9, 0x81c, 0x268, 0x768, 0x3a,  0x36f, 0x281, 0x4d1, 0x54a, 0x37e, 0x384, 0x2c5,
    0x54b, 0x48d, 0x9f1, 0x616, 0x7b,  0x55e, 0xa8,  0x35f, 0xc31, 0x73e, 0x693, 0xbbc, 0x89b, 0xa23, 0xc54, 0xc88,
    0x435, 0x537, 0x7eb, 0x207, 0x528, 0x80,  0x684, 0x39e, 0x805, 0x3e9, 0x2b6, 0x425, 0x62d, 0x1f8, 0x4b2, 0x44d,
    0x3a,  0x202, 0x5e6, 0x836, 0x111, 0x967, 0x3a9, 0x8d0, 0xc87, 0x59f, 0xb97, 0x871, 0x77b, 0x997, 0x97e, 0x726,
    0x240, 0x523, 0xa4,  0xc81, 0x66f, 0xb6a, 0x864, 0xcc,  0x4e5, 0xae9, 0x787, 0xc95, 0x7c7, 0x6b4, 0x5dd, 0x743,
    0x305, 0x92f, 0x7f7, 0x719, 0xa1,  0x649, 0xf3,  0x15a, 0xa36, 0x126, 0x606, 0x86f, 0x991, 0xcc8, 0x502, 0x6f5,
    0x691, 0x659, 0x1dc, 0x5e3, 0x86e, 0x2a9, 0x607, 0x3c4, 0x6dd, 0xd8,  0xbb6, 0x650, 0xac7, 0xa3,  0x119, 0xd00,
    0xbf5, 0xcfe, 0x5ed, 0x72e, 0x265, 0x26f, 0x4ff, 0x3b,  0x87f, 0x55e, 0x26c, 0xb60, 0x734, 0xc92, 0x5b9, 0x978,
    0xca9, 0xc2c, 0x6f9, 0xb55, 0x2de, 0xbb1, 0x983, 0x4d3, 0x14e, 0xa2c, 0x19a, 0x54d, 0x81e, 0x265, 0x738, 0x15b,
    0xc40, 0x8b5, 0x986, 0x306, 0xbc8, 0xb0f, 0xa7e, 0x300, 0x4f7, 0x491, 0x627, 0x4bb, 0x169, 0xca4, 0x936, 0xae2,
    0xc6e, 0x6c2, 0x853, 0x73d, 0x140, 0x5d8, 0x61b, 0x393, 0x64b, 0xc67, 0x440, 0x5ad, 0x7e6, 0x162, 0xcbb, 0x55f,
    0xb92, 0x43f, 0x245, 0x1f9, 0xa,   0x5b3, 0x45a, 0xe1,  0xafd, 0x4bb, 0x530, 0x59d, 0xa18, 0x6bd, 0x2b2, 0x41a,
    0xaa0, 0xa47, 0x7cf, 0x413, 0xc4,  0xb41, 0x46e, 0xa5d, 0xa3f, 0x792, 0x719, 0xd7,  0xa41, 0xcb9, 0x85d, 0x411,
    0x8c8, 0x97,  0xe4,  0x16d, 0x400, 0x86a, 0xb92, 0x611, 0x6a0, 0x803, 0x68e, 0xe4,  0xb2,  0x3eb, 0x5f4, 0x32b,
    0xc96, 0x9f1, 0x691, 0xc57, 0xa3,  0x5da, 0x394, 0x3aa, 0xbc7, 0x3b3, 0x990, 0xcb4, 0x4d9, 0x42e, 0x2a4, 0xb75,
    0xb95, 0x91b, 0xcf2, 0x2cc, 0xb1d, 0x512, 0x892, 0xa99, 0x4c0, 0x520, 0x5e1, 0x67c, 0x224, 0xc1b, 0x528, 0x280,
    0x2ad, 0xb3b, 0x35f, 0x7bb, 0xf6,  0xb20, 0xcbd, 0x1e6, 0xa40, 0x18b, 0x231, 0x37c, 0xb6e, 0x61a, 0x510, 0x75c,
    0x474, 0xb4e, 0x34e, 0x811, 0x75a, 0x8df, 0x1cd, 0x874, 0xc69, 0xb3d, 0x3d2, 0xa4f, 0xbbd, 0x2f1, 0x53a, 0x42a,
    0x346, 0x6a7, 0x235, 0xcc1, 0x652, 0x67d, 0xc9d, 0x81f, 0x8b6, 0x3ef, 0x297, 0x5c7, 0x4ae, 0x3dc, 0x9b7, 0x314,
    0xb2a, 0x9a1, 0x672, 0x641, 0x59e, 0x81b, 0xb4,  0xa2a, 0x66f, 0x72a, 0x31e, 0xc0b, 0xcb3, 0xb27, 0x31e, 0x34c,
    0xbfa, 0x2db, 0xcfd, 0x46a, 0x4db, 0xc4,  0x402, 0x262, 0xba6, 0x9a1, 0x863, 0x21d, 0x174, 0xbb3, 0xab6, 0x113,
    0x2ba, 0x301, 0x8c4, 0xb49, 0xbe,  0x990, 0xa44, 0x357, 0x984, 0xc18, 0x196, 0xcd0, 0x681, 0x728, 0x9b,  0xbaa,
    0x5f1, 0x827, 0x155, 0x40e, 0xae5, 0xa8e, 0x731, 0xb5a, 0x60e, 0x520, 0x954, 0x345, 0x63b, 0x843, 0x90a, 0x927,
    0x2d8, 0x81f, 0x524, 0x118, 0x832, 0x81,  0x182, 0x418, 0xa0d, 0x4cc, 0x8e,  0x53e, 0x898, 0x3d7, 0x634, 0xa72,
    0xc0b, 0x94f, 0xc1f, 0x3d3, 0x694, 0x7eb, 0x162, 0x878, 0xcff, 0xc6f, 0x1f9, 0xfd,  0x969, 0x4a,  0xb1f, 0x17c,
    0xacf, 0x8d8, 0xbc8, 0x54c, 0xbdb, 0x80f, 0x77f, 0xcdc, 0xc65, 0x2d8, 0x108, 0x737, 0xbe3, 0xbca, 0xa69, 0x71a,
  };
  // for(int i = 0; i < 2; ++i) {
  //   reference_ntt(reference_u_intt + i * 256);
  // }
  // for (int i = 0; i < 512; ++i) {
  //   ASSERT_EQ(reference_u_intt[i], reference_u_ntt[i]);
  // }

  Zq* d_u;
  cudaMalloc(&d_u, 512 * sizeof(int16_t));
  cudaMemcpy(d_u, reference_u_ntt, 512 * sizeof(int16_t), cudaMemcpyHostToDevice);
  inttInplaceKernel<2><<<1, 128>>>(d_u);
  int16_t h_u[512];
  cudaMemcpy(h_u, d_u, 512 * sizeof(int16_t), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 512; ++i) {
    printf("0x%03x, ", h_u[i]);
  }
  for (int i = 0; i < 512; ++i) {
    ASSERT_EQ(h_u[i], reference_u_intt[i]);
  }
}

TEST_F(KyberTest, NTT)
{
  constexpr uint16_t Q = 3329;
  constexpr uint8_t K = 4; // test a batch of 4 polys
  int16_t reference_arr[K][256];
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int> dist(1, Q - 1);
  for (int p = 0; p < K; p++)
    for (int i = 0; i < 256; i++)
      reference_arr[p][i] = int16_t(dist(rng));
  Zq host_arr[K * 256];
  for (int p = 0; p < K; p++)
    for (int i = 0; i < 256; i++)
      host_arr[p * 256 + i] = Zq(reference_arr[p][i]);

  Zq* d_A;
  cudaMalloc(&d_A, K * 256 * sizeof(Zq));
  cudaMemcpy(d_A, host_arr, K * 256 * sizeof(Zq), cudaMemcpyHostToDevice);
  nttInplaceKernel<K><<<1, 128>>>(d_A);
  Zq ntt_result[K * 256];
  cudaMemcpy(ntt_result, d_A, K * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);
  for (int p = 0; p < K; p++)
    reference_ntt(reference_arr[p]);
  for (int p = 0; p < K; p++)
    for (int i = 0; i < 256; i++)
      ASSERT_EQ(ntt_result[p * 256 + i], (uint32_t)reference_arr[p][i] % Q);

  inttInplaceKernel<K><<<1, 128>>>(d_A);
  Zq result[K * 256];
  cudaMemcpy(result, d_A, K * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);

  for (int p = 0; p < K; p++)
    for (int i = 0; i < 256; i++)
      ASSERT_EQ(result[p * 256 + i].raw(), host_arr[p * 256 + i].raw());

  cudaFree(d_A);
}

template <uint8_t k = 2>
__global__ void sampleA_test_kernel(const uint64_t rho[4], Zq* A)
{
  // generate_matrix_A<k>(rho, PolyMatrix<256, k, k, Zq>(A), );

  // const uint8_t warp_idx = threadIdx.x / 32;
  // if constexpr (k == 2){
  //   if (warp_idx < 2) {
  //     generate_matrix_A<k>(rho, PolyMatrix<256, k, k, Zq>(A), 2, warp_idx * 2);
  //   }
  // }else if constexpr (k == 3){
  //   if (warp_idx < 3) {
  //     generate_matrix_A<k>(rho, PolyMatrix<256, k, k, Zq>(A), 3, warp_idx * 3);
  //   }
  // }else if constexpr (k == 4){
  //   if (warp_idx < 3) {
  //     generate_matrix_A<k>(rho, PolyMatrix<256, k, k, Zq>(A), warp_idx == 0 ? 6 : 5, warp_idx * 5 + (warp_idx
  //     > 0));
  //   }
  // }
  switch (k) {
  case 2:
    generate_matrix_A<k, 0, 1>(rho, PolyMatrix<256, k, k, Zq>(A));
    break;
  case 3:
  case 4:
    generate_matrix_A<k, 0, 2>(rho, PolyMatrix<256, k, k, Zq>(A));
    break;
  default:
    __builtin_unreachable();
  }
}

TEST_F(KyberTest, SampleA)
{
  const std::string rho = "ec6e912ca1729d2ec0c233409baa3d6b105eb7f862e97ad86a367a8b798c325c";
  uint16_t A[2][2][256] = {
    {{1868, 3166, 2284, 1796, 255,  1512, 3309, 2171, 1237, 2788, 31,   398,  1224, 3230, 1480, 1629, 2356, 3295, 1736,
      970,  2214, 2887, 2903, 913,  641,  422,  578,  781,  2086, 2968, 653,  1057, 2430, 2194, 2022, 2881, 463,  766,
      641,  728,  3139, 3181, 1180, 182,  3257, 1708, 2899, 1483, 925,  2781, 3295, 1377, 1379, 2447, 1434, 3176, 2446,
      479,  380,  2698, 771,  117,  1233, 868,  96,   333,  1668, 835,  3171, 1587, 2396, 827,  1428, 715,  2014, 2796,
      3081, 2177, 811,  2294, 1600, 670,  799,  2526, 2825, 2989, 29,   2098, 313,  2496, 1732, 3212, 443,  1007, 1810,
      136,  1229, 310,  2405, 1700, 3237, 501,  3301, 1245, 2514, 297,  1261, 166,  1118, 1231, 763,  607,  2854, 114,
      3171, 2629, 505,  1605, 1302, 3118, 1660, 500,  524,  772,  1719, 1313, 2181, 1307, 509,  2202, 1788, 815,  3252,
      2424, 437,  1752, 3167, 557,  1043, 3084, 2499, 2046, 462,  1911, 3150, 2823, 2114, 2010, 2204, 1168, 1812, 411,
      1656, 2781, 2208, 3020, 1906, 941,  44,   55,   298,  3080, 657,  1299, 2111, 215,  1221, 2818, 539,  2781, 3153,
      645,  2279, 1467, 2104, 1046, 10,   2127, 80,   523,  2146, 840,  2017, 2416, 3059, 1741, 2800, 2988, 2361, 2963,
      2507, 1064, 928,  236,  1875, 6,    1850, 2639, 748,  3324, 1678, 2123, 1631, 1116, 3154, 1959, 739,  2354, 2405,
      3202, 2747, 2721, 2005, 386,  1165, 987,  3118, 289,  3275, 803,  1956, 770,  2906, 2754, 2122, 2713, 2266, 1006,
      2444, 2340, 1276, 681,  1389, 721,  3284, 739,  761,  1374, 1719, 698,  770,  1857, 3320, 2767, 1878, 79,   1196,
      740,  1325, 860,  2345, 1636, 1175, 1024, 689,  1480},
     {2672, 370,  480,  1622, 1418, 1637, 646,  2283, 2420, 1169, 813,  642,  593,  918,  1636, 641,  421,  1075, 2679,
      428,  160,  1186, 3163, 1536, 65,   870,  826,  1931, 2542, 2286, 832,  2824, 1879, 2723, 851,  959,  290,  1471,
      258,  1398, 1559, 3122, 2287, 953,  2455, 2417, 3053, 3174, 380,  2776, 1586, 1579, 1460, 2890, 3315, 167,  334,
      2506, 333,  1893, 1026, 1124, 1313, 1329, 1541, 1391, 1190, 1763, 214,  2527, 1742, 2454, 1512, 1329, 3252, 2013,
      1291, 1211, 331,  2463, 1378, 2123, 1291, 35,   1996, 2543, 1337, 2552, 1322, 1690, 2099, 1654, 393,  1280, 3263,
      2989, 2870, 1476, 676,  2252, 969,  2258, 729,  349,  1921, 770,  14,   687,  238,  1390, 88,   726,  499,  1846,
      2031, 3213, 2596, 865,  2309, 3048, 1333, 2771, 3171, 2962, 1132, 3167, 1905, 224,  365,  2351, 40,   469,  3136,
      934,  1210, 912,  284,  1922, 140,  2026, 942,  894,  2329, 3053, 1934, 2948, 2938, 628,  1188, 1652, 2456, 2136,
      427,  2287, 2060, 584,  1603, 3021, 2138, 1658, 2882, 2087, 2108, 2116, 1184, 375,  1861, 2976, 638,  761,  2946,
      2731, 2167, 343,  1293, 1626, 1491, 3167, 2390, 3220, 1799, 3312, 202,  1734, 672,  1591, 2858, 2535, 1625, 1294,
      1420, 3080, 443,  477,  3283, 445,  1546, 1213, 3212, 2417, 2557, 880,  1670, 2523, 1111, 2782, 185,  1625, 1768,
      705,  2858, 569,  3301, 189,  5,    2265, 487,  1014, 1887, 1557, 2053, 1711, 1088, 1773, 1614, 809,  1247, 2451,
      1419, 678,  1004, 445,  2196, 1822, 2909, 1853, 1170, 3188, 727,  217,  671,  2628, 1919, 2543, 2924, 561,  461,
      915,  1948, 2974, 96,   1530, 754,  826,  2093, 1084}},
    {{1381, 276,  305,  1225, 2773, 249,  1676, 3240, 1311, 138,  2679, 800,  761,  2658, 116,  3312, 2642, 861,  1937,
      2754, 609,  98,   2508, 1617, 2276, 2454, 2266, 2230, 2830, 3263, 2586, 491,  2330, 690,  2491, 2143, 2348, 179,
      2646, 1663, 2785, 790,  1229, 2738, 770,  2669, 1953, 2613, 2937, 1347, 1834, 2123, 2158, 726,  385,  2785, 867,
      200,  171,  553,  2445, 2029, 1315, 2330, 2589, 107,  1225, 115,  606,  1293, 1378, 2477, 2263, 1492, 2557, 746,
      1783, 2553, 1598, 1417, 3103, 2148, 525,  1136, 3315, 38,   402,  1872, 897,  45,   958,  2061, 2263, 1030, 2381,
      807,  2115, 2190, 2998, 1719, 2736, 1581, 2325, 2229, 1437, 1160, 87,   1022, 1547, 882,  1130, 2137, 1542, 2036,
      2060, 402,  3241, 1917, 253,  2634, 1942, 554,  774,  2031, 881,  1579, 888,  2073, 223,  187,  3327, 2229, 1800,
      2295, 29,   2111, 1932, 1871, 2376, 2346, 888,  1134, 1552, 2344, 2995, 2764, 1185, 2697, 1,    3056, 1282, 1471,
      3209, 1095, 464,  2435, 2877, 1316, 473,  1854, 2187, 1667, 2775, 866,  2086, 565,  1905, 1302, 2723, 3173, 1251,
      260,  1028, 1176, 819,  2588, 351,  595,  3241, 1122, 1884, 3086, 2204, 1277, 999,  3280, 2330, 2972, 2205, 1494,
      2659, 112,  338,  3174, 142,  840,  2697, 2005, 680,  1810, 715,  1690, 256,  1922, 3208, 2738, 2266, 289,  3249,
      1787, 1721, 521,  1422, 791,  2650, 2505, 2881, 797,  3252, 657,  2628, 2373, 1847, 3320, 894,  912,  130,  588,
      206,  3305, 559,  1051, 1233, 456,  993,  1926, 2221, 2252, 2275, 2759, 3146, 100,  1252, 286,  476,  2928, 2720,
      101,  2294, 2913, 268,  2857, 2569, 2557, 481,  991},
     {3263, 945,  898,  776,  403,  861,  1460, 2030, 2915, 2419, 600,  1217, 1926, 2988, 46,   2847, 201,  1081, 2067,
      2464, 2217, 1943, 1797, 1074, 3104, 2307, 1171, 1125, 521,  2612, 2694, 1270, 378,  225,  3239, 2894, 1995, 421,
      2672, 1602, 2497, 2486, 1729, 946,  2140, 3089, 3211, 407,  1767, 1370, 1461, 1143, 3196, 1078, 772,  2723, 1031,
      2191, 1024, 923,  1631, 2881, 1535, 1021, 2299, 2557, 2226, 128,  473,  2009, 3297, 2297, 1905, 2273, 3181, 2504,
      1021, 2044, 2901, 1032, 538,  2352, 450,  2184, 1642, 2209, 549,  1636, 315,  1314, 1098, 1021, 2487, 1792, 1133,
      2830, 1862, 1293, 2600, 1878, 467,  1519, 628,  2119, 631,  2529, 2301, 3213, 20,   1891, 2265, 2731, 1295, 2059,
      2555, 2807, 2449, 3025, 2478, 1315, 1117, 2412, 1780, 1285, 2085, 2695, 1286, 287,  2012, 997,  705,  3050, 1267,
      476,  235,  256,  2727, 643,  757,  1766, 103,  2890, 730,  1821, 2229, 1195, 1508, 264,  1041, 2670, 2350, 2992,
      2039, 3028, 97,   1906, 900,  1840, 3011, 2765, 3064, 1884, 688,  3326, 1417, 2307, 2996, 1983, 688,  1207, 284,
      818,  777,  646,  3034, 1389, 2223, 1081, 2469, 2012, 1103, 1078, 1631, 2208, 1014, 3087, 114,  2322, 1588, 692,
      8,    3295, 181,  3324, 1093, 2505, 1832, 2800, 2581, 3143, 276,  727,  477,  1220, 1001, 1534, 1224, 769,  3289,
      2626, 733,  2596, 1066, 95,   52,   1337, 3141, 2349, 2010, 383,  1815, 1666, 3143, 2617, 1418, 1017, 804,  345,
      2230, 2451, 2864, 248,  2399, 772,  1738, 2050, 2446, 3047, 3108, 2633, 1982, 1165, 3109, 832,  2330, 267,  1966,
      586,  1605, 1757, 2042, 1255, 2432, 2576, 1587, 244}}};
  // Allocate device memory
  uint64_t* d_rho;
  Zq* d_A;
  cudaMalloc(&d_rho, 4 * sizeof(uint64_t));
  cudaMalloc(&d_A, 4 * 256 * sizeof(Zq));

  // Convert hex string to bytes
  std::vector<uint8_t> rho_bytes(32);
  for (int i = 0; i < 32; i++) {
    rho_bytes[i] = std::stoi(rho.substr(i * 2, 2), nullptr, 16);
  }

  // Copy rho to device as uint64_t array
  uint64_t h_rho[4];
  memcpy(h_rho, rho_bytes.data(), 32);
  cudaMemcpy(d_rho, h_rho, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Run kernel
  sampleA_test_kernel<<<1, 128>>>(d_rho, d_A);

  // Copy results back
  uint16_t result[4][256];
  cudaMemcpy(result, d_A, 4 * 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

  printf("start verifying\n");

  // Verify results
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 256; k++) {
        ASSERT_EQ(A[i][j][k], result[i * 2 + j][k]) << "Mismatch at A[" << i << "][" << j << "][" << k << "]";
      }
    }
  }

  // Cleanup
  cudaFree(d_rho);
  cudaFree(d_A);
}

template <const uint8_t k = 2, const uint8_t eta = 3>
__global__ void samplePolyCBD_test_kernel(const uint64_t sigma[4], Zq* s_e)
{
  // generate_error_vector<k, 2 * k, eta, 0, false>(sigma, PolyVec<256, 2 * k, Zq>(s_e));

  switch (k) {
  case 2:
    generate_error_vector<k, 2 * k, eta, 0, false, 2, 3>(sigma, PolyVec<256, 2 * k, Zq>(s_e));
    break;
  case 3:
  case 4:
    generate_error_vector<k, 2 * k, eta, 0, false, 3, 3>(sigma, PolyVec<256, 2 * k, Zq>(s_e));
    break;
  default:
    __builtin_unreachable();
  }
}

TEST_F(KyberTest, SamplePolyCBD_3)
{
  const std::string sigma = "1a3b046be3aec19ef12d23ab6afbb2c2319f6764b35ca793422288211ad2b708";

  uint16_t s[2][256] = {
    {0,    1,    3328, 3328, 0,    3327, 3328, 0,    1,    0,    1,    3328, 0,    1,    1,    1,    3327, 1,    2,
     3328, 0,    3327, 3327, 0,    3328, 1,    3327, 1,    3328, 1,    3328, 3327, 0,    3327, 0,    3328, 1,    0,
     0,    3327, 3327, 0,    3328, 1,    1,    1,    3328, 1,    0,    0,    3328, 0,    3328, 3328, 2,    0,    2,
     1,    3327, 3328, 0,    0,    1,    2,    0,    2,    0,    1,    3327, 3328, 1,    0,    1,    3328, 0,    0,
     0,    1,    3327, 1,    0,    0,    3328, 0,    2,    1,    3327, 1,    1,    1,    1,    3328, 0,    3328, 0,
     0,    1,    1,    1,    1,    0,    3328, 2,    3327, 1,    3328, 0,    1,    3328, 3327, 3328, 3328, 2,    0,
     1,    3327, 0,    0,    0,    1,    1,    1,    0,    3328, 0,    3328, 3328, 0,    0,    0,    3326, 0,    2,
     3327, 3327, 3327, 0,    1,    3328, 1,    3328, 3328, 0,    2,    3327, 0,    2,    3328, 0,    3328, 0,    3328,
     0,    3328, 2,    3327, 2,    0,    1,    0,    1,    3327, 0,    0,    3328, 2,    0,    1,    3327, 3326, 3328,
     0,    3326, 3328, 2,    0,    0,    0,    3328, 3327, 1,    0,    1,    0,    3328, 3328, 3328, 3328, 2,    3327,
     2,    2,    1,    0,    0,    0,    0,    3327, 3327, 0,    0,    0,    0,    0,    3328, 3328, 1,    0,    3328,
     0,    1,    0,    1,    2,    0,    3327, 0,    3328, 1,    2,    0,    3327, 0,    3327, 3328, 0,    0,    0,
     0,    3327, 1,    3328, 3328, 1,    0,    0,    3328, 3327, 0,    3327, 3328, 3327, 1,    3327, 3327, 0,    1,
     3328, 1,    3328, 3328, 1,    3327, 2,    0,    1},
    {3328, 0,    0,    1,    0,    2,    2,    1,    3328, 0,    1,    0,    3328, 0, 2,    3328, 1,    3327, 1,
     3328, 1,    1,    0,    3328, 2,    0,    0,    0,    1,    3327, 1,    2,    2, 0,    0,    0,    1,    3327,
     1,    3327, 0,    2,    1,    3328, 2,    0,    3327, 2,    3328, 0,    0,    2, 0,    3328, 2,    0,    0,
     0,    0,    0,    0,    3327, 0,    1,    0,    1,    2,    0,    0,    3328, 2, 1,    3327, 1,    3328, 3327,
     0,    3328, 3327, 0,    0,    3328, 2,    1,    0,    0,    1,    3328, 0,    0, 3327, 0,    3328, 1,    2,
     3328, 3,    3328, 3327, 1,    0,    3328, 0,    3328, 3328, 0,    0,    0,    0, 0,    1,    2,    0,    1,
     0,    0,    1,    0,    0,    1,    0,    3328, 2,    2,    3328, 2,    0,    1, 0,    3328, 0,    0,    0,
     0,    3327, 3328, 3327, 3328, 2,    3327, 3328, 3,    0,    0,    3326, 3328, 1, 1,    0,    0,    0,    3328,
     0,    1,    3,    0,    3326, 1,    3328, 1,    3328, 1,    0,    3328, 3327, 1, 1,    0,    0,    3328, 0,
     1,    3328, 3328, 3328, 1,    3328, 1,    0,    0,    2,    3328, 0,    3328, 0, 0,    2,    3328, 0,    1,
     3328, 1,    0,    0,    0,    1,    0,    1,    2,    0,    1,    3328, 0,    0, 1,    1,    0,    1,    3328,
     1,    0,    1,    1,    0,    2,    0,    1,    1,    3328, 3328, 3,    0,    0, 1,    3327, 1,    0,    3328,
     3328, 3328, 0,    3327, 1,    1,    1,    2,    1,    0,    0,    3327, 3328, 1, 3327, 2,    3327, 3328, 0,
     3328, 3328, 1,    0,    3328, 2,    0,    2,    3327}};

  uint16_t e[2][256] = {
    {0,    3327, 1,    0,    1,    1,    0,    3328, 2,    0,    0,    0,    3328, 0,    3328, 3328, 0,    3328, 3328,
     0,    0,    0,    1,    3328, 0,    3328, 0,    0,    2,    1,    3327, 0,    0,    1,    3327, 3,    0,    0,
     1,    0,    3327, 3328, 0,    0,    3328, 1,    3328, 3,    0,    3328, 1,    3328, 1,    2,    1,    3328, 0,
     0,    3327, 1,    3328, 0,    3326, 0,    3327, 0,    3,    3328, 3328, 2,    3328, 3328, 1,    1,    3327, 3328,
     3,    3328, 1,    3328, 0,    3328, 3328, 3327, 3328, 2,    0,    0,    1,    1,    3,    3328, 0,    0,    0,
     1,    3328, 2,    3328, 3328, 0,    0,    1,    2,    0,    3328, 0,    1,    3328, 3328, 3327, 0,    1,    3328,
     3328, 0,    3328, 1,    1,    0,    0,    0,    2,    3328, 3328, 3327, 1,    1,    3327, 3328, 0,    3327, 3328,
     2,    3328, 0,    3328, 0,    1,    3328, 0,    3328, 3328, 3328, 1,    3328, 0,    3328, 0,    3,    1,    3328,
     3328, 0,    1,    3328, 0,    0,    1,    1,    1,    2,    0,    3328, 0,    3327, 0,    0,    3328, 0,    1,
     3328, 1,    0,    1,    3327, 1,    0,    0,    2,    0,    3327, 0,    0,    3328, 3328, 1,    2,    0,    2,
     3327, 0,    0,    1,    0,    3328, 1,    0,    3327, 3328, 3328, 1,    3328, 1,    1,    3328, 0,    1,    3328,
     0,    3328, 0,    0,    3327, 3328, 0,    3327, 3328, 3328, 3328, 1,    1,    3328, 1,    1,    1,    1,    1,
     3327, 0,    0,    0,    1,    1,    1,    3,    0,    0,    3328, 0,    3328, 1,    1,    3328, 2,    3328, 2,
     0,    1,    0,    3327, 0,    0,    1,    0,    0},
    {0,    3328, 0,    3328, 3328, 3328, 0,    3327, 0,    3327, 0,    0,    3328, 1,    3326, 3327, 1,    3328, 0,
     3328, 1,    0,    0,    3327, 3327, 3328, 1,    3327, 3328, 3327, 0,    2,    0,    3327, 3328, 0,    2,    3327,
     1,    0,    0,    3327, 3328, 0,    0,    3328, 2,    0,    2,    0,    0,    2,    2,    0,    1,    3328, 3327,
     2,    3328, 1,    0,    1,    0,    3328, 0,    1,    0,    3328, 2,    2,    1,    1,    0,    1,    0,    1,
     0,    3328, 1,    0,    0,    1,    0,    2,    3327, 3328, 0,    0,    3327, 3327, 0,    3328, 3328, 3328, 3,
     0,    0,    1,    0,    3328, 3327, 0,    2,    0,    3328, 0,    3328, 3328, 3327, 0,    2,    1,    1,    0,
     2,    0,    0,    3328, 3328, 0,    3327, 0,    3328, 1,    3328, 0,    2,    3328, 1,    1,    3328, 0,    1,
     0,    0,    3328, 0,    1,    1,    3328, 0,    0,    1,    0,    2,    0,    2,    1,    0,    0,    3328, 0,
     0,    3327, 0,    0,    1,    3327, 0,    1,    2,    0,    2,    0,    0,    1,    0,    3328, 1,    3328, 0,
     1,    0,    3328, 3328, 3327, 0,    1,    1,    0,    0,    3328, 3327, 0,    3328, 0,    0,    3327, 0,    1,
     3328, 3328, 1,    1,    2,    2,    0,    3,    0,    0,    1,    3327, 3328, 0,    0,    0,    3326, 3327, 2,
     3328, 3328, 3328, 2,    1,    0,    2,    3328, 3328, 1,    0,    3328, 2,    2,    3328, 3328, 1,    3328, 3328,
     0,    0,    1,    0,    2,    1,    2,    0,    3328, 1,    3328, 0,    1,    0,    0,    0,    3328, 3328, 2,
     1,    3328, 0,    1,    1,    0,    1,    1,    1}};

  // Convert sigma hex string to bytes
  std::array<uint8_t, 32> sigma_bytes;
  for (int i = 0; i < 32; i++) {
    sigma_bytes[i] = std::stoi(sigma.substr(i * 2, 2), nullptr, 16);
  }

  // Copy sigma to device as uint64_t array
  uint64_t h_sigma[4];
  memcpy(h_sigma, sigma_bytes.data(), 32);
  uint64_t* d_sigma;
  cudaMalloc((void**)&d_sigma, 4 * sizeof(uint64_t));
  cudaMemcpy(d_sigma, h_sigma, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Allocate device memory for s and e as one contiguous array
  Zq* d_se;
  cudaMalloc((void**)&d_se, 4 * 256 * sizeof(Zq)); // Allocate space for both s and e
  Zq* d_s = d_se;                                  // First half is s
  Zq* d_e = d_se + (2 * 256);                      // Second half is e

  // Run kernel to sample s and e
  samplePolyCBD_test_kernel<<<1, 128>>>(d_sigma, d_se);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Copy results back
  uint16_t result_s[2][256];
  uint16_t result_e[2][256];
  cudaMemcpy(result_s, d_s, 2 * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);
  cudaMemcpy(result_e, d_e, 2 * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 256; j++) {
      ASSERT_EQ(s[i][j], result_s[i][j]) << "Mismatch at s[" << i << "][" << j << "]";
      ASSERT_EQ(e[i][j], result_e[i][j]) << "Mismatch at e[" << i << "][" << j << "]";
    }
  }

  // Cleanup
  cudaFree(d_sigma);
  cudaFree(d_se); // Only need to free the base pointer
}

__global__ void test_sum_3_bits_and_subtract_kernel(const uint64_t* input, uint64_t* output)
{
  output[threadIdx.x] = sum_3_bits_and_subtract(input[threadIdx.x]);
}

__global__ void test_sum_3_bits_and_subtract_8_kernel(const uint8_t* input, uint8_t* output)
{
  output[threadIdx.x] = sum_3_bits_and_subtract_8(input[threadIdx.x]);
}

TEST_F(KyberTest, Sum3BitsAndSubtract)
{
  uint64_t test_inputs[] = {0, 0x249249249249249, 0x3333333333333333};
  uint64_t expected_outputs[] = {0x0104104104104104, 0x0104104104104104, 0x0104104104104104};

  uint64_t* d_input;
  uint64_t* d_output;
  cudaMalloc(&d_input, sizeof(test_inputs));
  cudaMalloc(&d_output, sizeof(test_inputs));
  cudaMemcpy(d_input, test_inputs, sizeof(test_inputs), cudaMemcpyHostToDevice);

  test_sum_3_bits_and_subtract_kernel<<<1, 3>>>(d_input, d_output);

  uint64_t results[3];
  cudaMemcpy(results, d_output, sizeof(results), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 3; ++i)
    ASSERT_EQ(results[i], expected_outputs[i]);

  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_F(KyberTest, Sum3BitsAndSubtract8)
{
  uint8_t test_inputs[] = {0, 0x49, 0x33};
  uint8_t expected_outputs[] = {4, 4, 4};

  uint8_t* d_input;
  uint8_t* d_output;
  cudaMalloc(&d_input, sizeof(test_inputs));
  cudaMalloc(&d_output, sizeof(test_inputs));
  cudaMemcpy(d_input, test_inputs, sizeof(test_inputs), cudaMemcpyHostToDevice);

  test_sum_3_bits_and_subtract_8_kernel<<<1, 3>>>(d_input, d_output);

  uint8_t results[3];
  cudaMemcpy(results, d_output, sizeof(results), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 3; ++i)
    ASSERT_EQ(results[i], expected_outputs[i]);

  cudaFree(d_input);
  cudaFree(d_output);
}

__device__ inline void
base_case_multiply(const Zq& a0, const Zq& a1, const Zq& b0, const Zq& b1, const Zq& gamma, Zq& out0, Zq& out1)
{
  out0 = a0 * b0 + a1 * b1 * gamma;
  out1 = a0 * b1 + a1 * b0;
}

// __device__ __constant__ Zq d_gamma[128] = {
//       17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229,
//       1409, 1920, 2662, 667, 3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279,
//       1703, 1626, 1651, 1678, 2789, 540, 1789, 1540, 1847, 1482, 952, 2377, 1461, 1868, 2687,
//       642, 939, 2390, 2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268, 3061, 641,
//       2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239,
//       1645, 1684, 1063, 2266, 319, 3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466,
//       863, 2594, 735, 2804, 525, 1092, 2237, 403, 2926, 1026, 2303, 1143, 2186, 2150, 1179,
//       2775, 554, 886, 2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300, 2110, 1219, 2935, 394, 885, 2444, 2154,
//       1175
//     };

// Algorithm 11: MultiplyNTTs
template <typename T>
__device__ inline void ntt_multiply(const Poly<256, T>& a, const Poly<256, T>& b, Poly<256, T>& out)
{
  if (threadIdx.x >= 128) { return; }
  base_case_multiply(
    a[threadIdx.x * 2], a[threadIdx.x * 2 + 1], b[threadIdx.x * 2], b[threadIdx.x * 2 + 1], d_gamma[threadIdx.x],
    out[threadIdx.x * 2], out[threadIdx.x * 2 + 1]);
}

template <typename T>
__global__ void ntt_multiply_kernel(const T* a, const T* b, T* out)
{
  Poly<256, T> a_poly(const_cast<T*>(a));
  Poly<256, T> b_poly(const_cast<T*>(b));
  Poly<256, T> out_poly(out);
  ntt_multiply(a_poly, b_poly, out_poly);
}

template <typename T, uint8_t COLS, uint8_t ROWS>
__device__ inline void transposed_matrix_vec_mult(
  const PolyMatrix<256, COLS, ROWS, T>& A, const PolyVec<256, COLS, T>& x, PolyVec<256, ROWS, T>& y)
{
  __shared__ T temp[256];
  Poly<256, T> temp_poly(temp);

  // each of the 128 threads does one pair (2 coefficients)
  if (threadIdx.x >= 128) return;

  for (uint8_t i = 0; i < ROWS; ++i) {
    // zero out this thread's two slots
    const int base = threadIdx.x * 2;
    y[i][base] = 0;
    y[i][base + 1] = 0;
    __syncthreads();

    for (uint8_t j = 0; j < COLS; ++j) {
      ntt_multiply(A[j][i], x[j], temp_poly);
      __syncthreads();

      // accumulate *your* two results
      y[i][base] += temp_poly[base];
      y[i][base + 1] += temp_poly[base + 1];
      __syncthreads();
    }
  }
}

template <typename T, uint8_t COLS, uint8_t ROWS>
__global__ void transposed_matrix_vec_mult_kernel(T* A, T* x, T* y)
{
  PolyMatrix<256, COLS, ROWS, T> A_matrix(A);
  PolyVec<256, COLS, T> x_vec(x);
  PolyVec<256, ROWS, T> y_vec(y);
  transposed_matrix_vec_mult(A_matrix, x_vec, y_vec);
}

template <uint8_t k>
__global__ void test_matrix_vec_mult(Zq* A, Zq* x, Zq* output)
{
  PolyMatrix<256, k, k, Zq> A_matrix(A);
  PolyVec<256, k, Zq> x_vec(x);
  PolyVec<256, k, Zq> y_vec(output);
  matrix_vec_mult<true, false, k>(A_matrix, x_vec, y_vec);
}

// TEST_F(KyberTest, TransposedMatrixVecMult) {

//   constexpr uint16_t Q = 3329;
//   constexpr uint8_t K = 4; // test a batch of 4 polys
//   reference_poly reference_matrix[K][K];
//   reference_poly reference_vec[K];
//   std::mt19937_64 rng(42);
//   std::uniform_int_distribution<int> dist(1, Q - 1);
//   for (int row = 0; row < K; row++) {
//     for (int col = 0; col < K; col++) {
//       for (int i = 0; i < 256; i++) {
//         reference_matrix[row][col].coeffs[i] = dist(rng);
//       }
//     }
//   }
//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < 256; j++) {
//       reference_vec[i].coeffs[j] = dist(rng);
//     }
//   }
//   reference_poly reference_result[K] = {0};
//   for (int i = 0; i < K; i++) {
//     reference_poly temp;
//     for (int j = 0; j < K; j++) {
//       reference_poly_basemul_montgomery(&temp, &reference_matrix[j][i], &reference_vec[j]);
//     }
//     for (int j = 0; j < 256; j++) {
//       reference_result[i].coeffs[j] = (reference_result[i].coeffs[j] + temp.coeffs[j]) % 3329;
//     }
//   }
//   Zq internal_A[256 * 4 * 4] = {0};
//   PolyMatrix<256, 4, 4, Zq> host_A(internal_A);
//   for (int row = 0; row < K; row++) {
//     for (int col = 0; col < K; col++) {
//       for (int i = 0; i < 256; i++) {
//         Zq zq_coeff = Zq(reference_matrix[row][col].coeffs[i]);
//         PolyVec<256, 4, Zq> host_A_row = host_A[row];
//         Poly<256, Zq> host_A_row_col = host_A_row[col];
//         host_A_row_col[i] = zq_coeff;
//       }
//     }
//   }
//   Zq internal_x[256 * 4] = {0};
//   PolyVec<256, 4, Zq> host_x(internal_x);
//   Zq internal_y[256 * 4] = {0};
//   PolyVec<256, 4, Zq> host_y(internal_y);
//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < 256; j++) {
//       host_x[i][j] = Zq(reference_vec[i].coeffs[j]);
//     }
//   }
//   Zq* d_A;
//   Zq* d_x;
//   Zq* d_y;
//   cudaMalloc(&d_A, PolyMatrix<256, 4, 4, Zq>::byte_size());
//   cudaMalloc(&d_x, PolyVec<256, 4, Zq>::byte_size());
//   cudaMalloc(&d_y, PolyVec<256, 4, Zq>::byte_size());
//   cudaMemcpy(d_A, host_A.data(), PolyMatrix<256, 4, 4, Zq>::byte_size(), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_x, host_x.data(), PolyVec<256, 4, Zq>::byte_size(), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_y, host_y.data(), PolyVec<256, 4, Zq>::byte_size(), cudaMemcpyHostToDevice);
//   transposed_matrix_vec_mult_kernel<Zq, 4, 4><<<1, 128>>>(d_A, d_x, d_y);
//   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA error: " << cudaGetErrorString(cudaGetLastError());
//   cudaMemcpy(host_y.data(), d_y, PolyVec<256, 4, Zq>::byte_size(), cudaMemcpyDeviceToHost);
//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < 256; j++) {
//       ASSERT_EQ(host_y[i][j], Zq(reference_result[i].coeffs[j]));
//     }
//   }
// }
TEST_F(KyberTest, NTTMultiply)
{
  const uint16_t f_ntt[256] = {
    2402, 2570, 2524, 2417, 1638, 1722, 2570, 3236, 207,  2642, 2042, 1466, 1918, 2389, 71,   2771, 2818, 1457, 210,
    305,  1446, 667,  922,  70,   1138, 3246, 539,  2538, 1802, 2850, 1128, 2290, 921,  673,  207,  1177, 1326, 2261,
    919,  418,  936,  3075, 2242, 2387, 3055, 920,  2979, 2590, 2648, 1477, 1398, 3129, 1542, 1940, 1583, 2554, 557,
    1054, 2378, 1191, 176,  208,  2136, 3150, 810,  3076, 2676, 2389, 578,  741,  1008, 1928, 740,  1887, 2190, 2988,
    3113, 2620, 811,  670,  3140, 251,  1784, 1817, 2770, 775,  3186, 1412, 2168, 1122, 1041, 3123, 2945, 3134, 126,
    1175, 1061, 5,    2415, 1843, 3003, 3124, 1759, 2214, 2126, 338,  2948, 2242, 2679, 815,  46,   1435, 957,  906,
    1332, 2973, 2761, 1229, 2804, 137,  1355, 1998, 2404, 298,  2194, 2641, 2150, 1842, 2178, 2522, 2708, 885,  2833,
    1307, 3268, 2723, 1206, 1063, 1597, 1533, 922,  1758, 495,  752,  2869, 189,  165,  2451, 2902, 2434, 979,  245,
    2447, 1214, 2913, 3082, 3050, 3161, 807,  1269, 3180, 2694, 915,  1341, 873,  1962, 1871, 3230, 950,  948,  2721,
    2485, 270,  1623, 763,  891,  2408, 3291, 1365, 2785, 3115, 641,  2007, 1433, 1477, 2733, 3079, 1373, 601,  845,
    607,  286,  167,  3204, 2402, 851,  2700, 433,  1585, 1630, 753,  727,  399,  2012, 382,  2452, 1306, 962,  3114,
    2512, 2359, 2159, 2146, 323,  3190, 2599, 324,  2032, 2733, 1244, 3018, 520,  985,  701,  1763, 2779, 962,  2570,
    1458, 2048, 2029, 2236, 1074, 3203, 639,  644,  1300, 1061, 2997, 3203, 891,  122,  732,  2429, 184,  2884, 1298,
    411,  2093, 3237, 2413, 2797, 1282, 751,  2996, 874,
  };

  const uint16_t g_ntt[256] = {
    3103, 63,   610,  1077, 2814, 3326, 1472, 1886, 915,  1096, 2736, 2279, 1800, 914,  1477, 2654, 2426, 810,  2843,
    2864, 544,  1795, 3136, 2393, 558,  482,  1060, 611,  3121, 2942, 84,   1456, 1928, 294,  2198, 2612, 3269, 1968,
    390,  2454, 880,  3068, 2291, 1331, 2126, 696,  3327, 1183, 2165, 3271, 2693, 452,  1860, 2726, 2250, 2823, 2842,
    723,  1546, 2142, 509,  970,  3136, 2047, 363,  2799, 1154, 3122, 863,  913,  162,  1413, 3251, 89,   702,  1053,
    2023, 275,  753,  2471, 1920, 3236, 808,  1874, 2863, 3200, 730,  659,  2367, 190,  3115, 809,  176,  236,  704,
    2138, 1981, 2077, 2565, 1160, 2298, 1847, 2532, 2295, 1201, 2413, 2776, 2416, 3117, 1622, 2446, 581,  507,  1638,
    2027, 124,  204,  1774, 131,  3131, 912,  2082, 2316, 2875, 3003, 2155, 1646, 496,  142,  660,  2573, 3282, 2912,
    505,  2744, 2609, 2029, 1267, 819,  1347, 2604, 589,  136,  2122, 2094, 3211, 348,  152,  268,  3176, 2260, 2291,
    1795, 1965, 2095, 2660, 3305, 436,  2433, 1185, 144,  1600, 2212, 1257, 2685, 2320, 2505, 647,  1132, 2407, 2400,
    1923, 2293, 1512, 2482, 1331, 3121, 3261, 65,   931,  326,  3100, 1220, 183,  183,  773,  3288, 1493, 529,  2887,
    1288, 803,  1372, 2732, 2321, 2997, 1945, 2037, 566,  3202, 1030, 505,  1938, 1707, 2288, 312,  943,  3029, 1112,
    2494, 2087, 493,  3089, 701,  1389, 54,   2288, 1850, 279,  1804, 533,  1797, 607,  1369, 1462, 1619, 633,  672,
    304,  2106, 844,  1781, 1045, 1249, 2279, 534,  2806, 1540, 475,  353,  2458, 2730, 978,  1344, 1165, 3255, 2649,
    907,  2950, 2951, 890,  2588, 783,  646,  664,  2494,
  };

  const uint16_t h_expected[256] = {
    2491, 3276, 1186, 1507, 106,  428,  2215, 2918, 563,  1076, 904,  2636, 2843, 1130, 395,  107,  1777, 1499, 875,
    466,  350,  2266, 1110, 2354, 2950, 2852, 839,  206,  424,  1478, 39,   449,  1784, 359,  1158, 1799, 1806, 461,
    2735, 1392, 1829, 1573, 2270, 388,  2147, 846,  3069, 224,  2581, 1415, 1074, 84,   227,  2058, 437,  1937, 2393,
    2599, 2868, 655,  2051, 285,  1582, 2672, 264,  1514, 1041, 2505, 2709, 2047, 2276, 2231, 1600, 1899, 896,  2708,
    1476, 1014, 1375, 1754, 2668, 147,  1360, 947,  742,  584,  2093, 1074, 3065, 1685, 2032, 739,  2290, 1558, 1820,
    1347, 2116, 3146, 111,  1826, 2758, 2055, 2941, 1969, 3137, 3178, 1738, 199,  2624, 1321, 1864, 1338, 991,  2876,
    3146, 2828, 1305, 2096, 449,  2053, 1255, 2660, 364,  1561, 1158, 2135, 3070, 333,  1398, 1273, 2328, 2624, 1884,
    132,  545,  2279, 2359, 2955, 2474, 1119, 1737, 888,  3244, 828,  1368, 631,  1970, 2501, 415,  1908, 2078, 229,
    516,  3243, 3255, 527,  233,  2232, 2571, 2366, 1343, 3060, 2067, 1803, 787,  2820, 2457, 461,  1191, 825,  2918,
    1056, 1181, 1819, 2114, 1214, 2690, 623,  563,  396,  989,  1639, 814,  1626, 2518, 663,  567,  3227, 826,  1597,
    1186, 236,  299,  1779, 3249, 2570, 2345, 340,  784,  2221, 1262, 544,  2212, 2974, 1493, 151,  2194, 2700, 1152,
    72,   1667, 2862, 748,  2014, 0,    527,  1242, 2112, 2027, 943,  2214, 1258, 1818, 2944, 5,    2862, 2338, 2896,
    1011, 1279, 1673, 1325, 1020, 1329, 1514, 1255, 2039, 2311, 2862, 2720, 967,  2526, 1744, 409,  3326, 599,  1532,
    2305, 2539, 2726, 1945, 2207, 3200, 1380, 2475, 2838,
  };

  Zq host_f[256];
  Zq host_g[256];
  for (int i = 0; i < 256; i++) {
    host_f[i] = Zq(f_ntt[i]);
    host_g[i] = Zq(g_ntt[i]);
  }
  Zq host_result[256];

  Zq* d_f;
  Zq* d_g;
  Zq* d_result;
  cudaMalloc(&d_f, 256 * sizeof(Zq));
  cudaMalloc(&d_g, 256 * sizeof(Zq));
  cudaMalloc(&d_result, 256 * sizeof(Zq));
  cudaMemcpy(d_f, host_f, 256 * sizeof(Zq), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, host_g, 256 * sizeof(Zq), cudaMemcpyHostToDevice);
  ntt_multiply_kernel<Zq><<<1, 128>>>(d_f, d_g, d_result);
  cudaMemcpy(host_result, d_result, 256 * sizeof(Zq), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 256; i++) {
    ASSERT_EQ(host_result[i].raw(), h_expected[i]);
  }
}

TEST_F(KyberTest, TransposedMatrixVecMult)
{
  constexpr uint8_t K = 4;

  const uint16_t reference_matrix[4][4][256] = {
    {
      {646,  2769, 1148, 3268, 1280, 885,  767,  3281, 1168, 2062, 554,  716,  1535, 1765, 362,  1139, 1546, 1133, 2047,
       1202, 2847, 678,  1676, 2274, 3096, 3226, 2790, 3328, 3122, 59,   1457, 1329, 3251, 637,  448,  2895, 1496, 3044,
       1867, 1325, 2777, 620,  696,  254,  301,  986,  1483, 2875, 264,  1847, 1193, 2720, 2680, 1478, 354,  679,  3241,
       3218, 2356, 3048, 1583, 1936, 1436, 2725, 2139, 373,  2177, 576,  523,  371,  1690, 1483, 2491, 268,  2780, 1336,
       1648, 2853, 3161, 1575, 414,  1287, 2061, 2305, 278,  1622, 309,  2051, 2199, 1008, 2709, 2153, 3104, 2800, 2071,
       2815, 23,   1250, 2504, 433,  2158, 3153, 3170, 1029, 897,  433,  2940, 1862, 3010, 2476, 2657, 942,  3277, 802,
       2249, 677,  1016, 738,  1205, 1953, 1760, 951,  84,   2373, 1397, 94,   241,  2579, 1737, 1737, 1670, 401,  443,
       2584, 1208, 1606, 1271, 3044, 2683, 2697, 1795, 315,  2665, 350,  2842, 3084, 1549, 771,  2352, 563,  2219, 2730,
       725,  2356, 173,  2227, 458,  1179, 1276, 134,  363,  2360, 2464, 3052, 516,  1231, 677,  2,    1913, 2904, 1002,
       1526, 587,  1091, 3166, 2520, 844,  2345, 1033, 168,  2004, 1246, 1642, 3131, 2066, 609,  516,  1703, 1654, 824,
       3024, 1255, 2594, 613,  1758, 236,  715,  2805, 994,  1905, 2549, 345,  910,  315,  753,  1747, 2630, 1399, 3307,
       1184, 2330, 2908, 929,  1580, 1914, 262,  67,   2902, 667,  2153, 20,   1520, 1276, 1146, 191,  244,  2484, 1576,
       1486, 2015, 2975, 1349, 1960, 634,  2169, 1799, 3019, 2736, 2828, 85,   637,  261,  2685, 1112, 2673, 2132, 233,
       3129, 2098, 2758, 2652, 389,  3021, 1385, 1249, 226},
      {2200, 2811, 2827, 2648, 55,   1826, 2330, 932,  2462, 2936, 2431, 2748, 26,   955,  1017, 3190, 3121, 2171, 1948,
       2373, 1959, 863,  3011, 253,  1654, 2301, 1184, 1667, 2738, 840,  3227, 2687, 2308, 1907, 428,  1907, 2293, 614,
       755,  2917, 1687, 1729, 1978, 1281, 861,  3066, 2006, 1971, 710,  1047, 838,  2200, 1324, 1418, 1148, 1934, 1131,
       1398, 1447, 2348, 3001, 1295, 561,  146,  1089, 2807, 1194, 2868, 1093, 380,  3277, 1894, 1256, 1094, 410,  1372,
       2133, 2391, 2072, 2920, 274,  2480, 3142, 1176, 3146, 609,  942,  2986, 318,  14,   1101, 2382, 2171, 2443, 106,
       327,  2387, 2189, 959,  2123, 1872, 1366, 3189, 778,  1412, 2842, 493,  860,  405,  297,  794,  823,  1196, 1652,
       2503, 2374, 3183, 3196, 2153, 1781, 2914, 1264, 2090, 977,  45,   2715, 1775, 1023, 1179, 321,  1872, 1134, 2648,
       2820, 608,  3315, 1714, 1029, 2260, 1181, 2593, 2978, 1665, 1265, 1841, 217,  146,  1478, 1780, 1780, 2415, 318,
       1044, 158,  2063, 1784, 771,  1326, 888,  857,  1871, 2261, 308,  1652, 2032, 2524, 3222, 1450, 296,  784,  1342,
       1507, 1354, 1783, 729,  1381, 1101, 631,  3043, 32,   2269, 2833, 347,  552,  816,  2364, 688,  2856, 2193, 588,
       1250, 2439, 732,  3225, 149,  1406, 375,  75,   2753, 1529, 1385, 3205, 1931, 3124, 15,   157,  2033, 1953, 188,
       1106, 602,  1287, 2573, 2723, 2986, 3188, 2931, 905,  1605, 259,  2163, 2394, 1883, 2201, 1766, 3291, 1366, 3153,
       2989, 1943, 2806, 659,  1327, 1238, 76,   2495, 589,  867,  1670, 2263, 2685, 361,  1527, 757,  3054, 2565, 219,
       1553, 512,  569,  1626, 3218, 2067, 1713, 2389, 3068},
      {3094, 2976, 2926, 882,  2828, 906,  828,  2618, 3101, 1708, 2387, 1860, 458,  3121, 2290, 2796, 1552, 836,  2859,
       2569, 1341, 2594, 1910, 2029, 1776, 1917, 931,  152,  1454, 2877, 1197, 1980, 2788, 2699, 1426, 2592, 670,  2312,
       3219, 3242, 943,  1987, 1946, 2111, 2343, 1412, 717,  2750, 1900, 1958, 631,  1450, 2560, 2136, 2424, 856,  542,
       1479, 3215, 2190, 2344, 772,  3300, 1896, 1600, 1,    3053, 1263, 1803, 3248, 1304, 2582, 1039, 275,  2296, 571,
       1905, 2221, 2335, 220,  1360, 88,   362,  1311, 513,  3015, 833,  265,  2441, 2122, 2534, 2951, 2734, 2395, 2100,
       2517, 3015, 2294, 1538, 2374, 1498, 2043, 2060, 256,  765,  340,  1462, 2014, 1228, 981,  222,  77,   2899, 697,
       1355, 2024, 1675, 1172, 1147, 145,  1310, 1437, 476,  2311, 1401, 1145, 275,  923,  1571, 1567, 2864, 1279, 2221,
       1073, 2955, 1192, 920,  2586, 2972, 136,  65,   151,  2533, 2394, 276,  2914, 2717, 1100, 1670, 1164, 2538, 2814,
       906,  137,  1919, 657,  1672, 2489, 2607, 3223, 749,  2674, 2310, 1118, 2528, 121,  1791, 1224, 2658, 2004, 2923,
       1726, 927,  2170, 2830, 276,  2086, 2784, 3241, 804,  1923, 2649, 2608, 2371, 3028, 1316, 1277, 1951, 1876, 3294,
       398,  3120, 3028, 716,  1921, 1388, 1568, 1521, 775,  1222, 2814, 1748, 481,  292,  1022, 3231, 2859, 3090, 3161,
       2723, 1256, 2338, 3191, 750,  2039, 841,  1440, 53,   166,  596,  1044, 2548, 356,  554,  2673, 441,  3156, 520,
       1171, 3088, 2256, 1864, 2624, 3191, 280,  911,  879,  1996, 3223, 2345, 862,  1104, 3042, 2681, 608,  1042, 1860,
       3225, 1781, 249,  36,   2943, 131,  2641, 2429, 1134},
      {3067, 2632, 2358, 1821, 2255, 2584, 1502, 1696, 1530, 1584, 3072, 3123, 491,  50,   1461, 2054, 3,    521,  3293,
       1728, 1235, 286,  1437, 753,  3300, 1912, 2930, 389,  2174, 948,  105,  692,  3237, 1898, 1749, 1202, 1634, 41,
       420,  316,  801,  2000, 875,  2851, 3045, 374,  2184, 2057, 1096, 2607, 1831, 1389, 3163, 1323, 2958, 2198, 1450,
       2374, 1385, 90,   1881, 401,  2242, 3197, 496,  3051, 2243, 2177, 2735, 1175, 3025, 335,  1285, 2791, 3203, 2116,
       703,  307,  2384, 745,  1833, 2636, 1977, 2209, 3292, 2786, 2395, 1659, 3042, 2857, 1367, 2561, 2045, 2793, 1541,
       1571, 667,  1117, 3106, 3307, 1621, 2036, 2045, 669,  379,  184,  561,  2215, 797,  1786, 2656, 432,  596,  1772,
       3268, 1366, 920,  2746, 1163, 2647, 3205, 3174, 2858, 2801, 42,   2279, 2599, 1708, 1926, 1470, 3310, 1415, 334,
       3116, 146,  2786, 820,  2545, 1856, 217,  1852, 1166, 439,  777,  162,  909,  1382, 1551, 2277, 1140, 2420, 1516,
       1717, 2559, 2542, 908,  2165, 289,  792,  634,  2099, 2831, 119,  1786, 784,  3322, 1167, 1610, 2466, 2105, 210,
       1972, 47,   3075, 2624, 985,  649,  2731, 1680, 2810, 522,  11,   1236, 680,  1214, 2943, 3096, 2575, 999,  2438,
       708,  830,  2112, 1813, 2244, 1224, 2484, 2829, 289,  3059, 2017, 223,  2378, 2889, 596,  279,  257,  1890, 2233,
       3034, 3223, 211,  1881, 3301, 124,  2629, 671,  1302, 1392, 1118, 1092, 555,  2662, 1246, 263,  796,  1745, 2842,
       792,  175,  3135, 1445, 3111, 2223, 1051, 421,  2176, 1698, 616,  2558, 1470, 528,  1805, 3272, 2929, 1737, 1066,
       3248, 2848, 3015, 3033, 3232, 518,  1562, 2665, 1091},
    },
    {
      {1751, 175,  1815, 1988, 2745, 2425, 2171, 2424, 2818, 1336, 222,  2242, 2769, 198,  2499, 3225, 3313, 3208, 989,
       2295, 1443, 1865, 402,  1815, 740,  3188, 1297, 417,  184,  702,  1457, 3145, 547,  2645, 1767, 3247, 191,  766,
       2424, 929,  2535, 902,  3251, 3028, 1832, 2953, 539,  2898, 2511, 2244, 1444, 1973, 1646, 2460, 2225, 1343, 2374,
       1076, 1925, 1140, 388,  136,  2534, 2999, 2936, 668,  2813, 515,  3321, 915,  1323, 23,   582,  1043, 466,  1931,
       2614, 353,  1616, 1072, 2949, 2115, 3246, 2895, 399,  705,  562,  1232, 1790, 1092, 3268, 1589, 1638, 1442, 2535,
       2095, 326,  2734, 28,   2630, 1894, 2686, 1270, 59,   1841, 1950, 1977, 377,  2478, 954,  1081, 1133, 1595, 2241,
       2910, 3267, 1430, 2573, 1293, 1683, 2501, 408,  769,  1817, 3050, 1566, 3234, 2705, 2751, 698,  2455, 178,  1478,
       1503, 3232, 2571, 2101, 3091, 2706, 3099, 1744, 2133, 1480, 1965, 2820, 764,  952,  1543, 2148, 23,   2096, 852,
       1655, 1212, 1270, 1738, 6,    1573, 1454, 489,  2947, 2216, 2694, 721,  1839, 2583, 1473, 2226, 1089, 3047, 764,
       167,  1792, 617,  1715, 608,  2142, 1120, 2814, 1346, 2348, 1565, 172,  2452, 3041, 1732, 1647, 2567, 263,  894,
       515,  2271, 16,   850,  1617, 1823, 2866, 1584, 1611, 1162, 1054, 1822, 1529, 2248, 2157, 2148, 2858, 1169, 2915,
       2622, 145,  2076, 791,  1543, 346,  170,  1435, 3128, 1791, 949,  991,  1285, 3242, 166,  3280, 3286, 2401, 2143,
       2275, 2085, 833,  1516, 1908, 2160, 2344, 2708, 1679, 2057, 836,  2946, 152,  298,  1126, 1997, 0,    1029, 2561,
       2762, 1975, 1693, 441,  2354, 916,  353,  3069, 2623},
      {2577, 1484, 2592, 819,  1006, 2104, 201,  3041, 303,  3164, 2684, 2377, 1623, 1200, 2856, 2191, 993,  3309, 1245,
       2031, 1199, 3308, 659,  3043, 1967, 802,  2514, 2934, 532,  2323, 947,  2908, 1280, 1722, 2192, 284,  1901, 874,
       2451, 2506, 959,  1151, 1946, 3050, 2208, 2404, 1877, 896,  1488, 310,  1091, 665,  3199, 2027, 1194, 1116, 2995,
       3049, 1752, 762,  2874, 1425, 412,  144,  2102, 533,  181,  2710, 1753, 2183, 1767, 1976, 3248, 2123, 1445, 411,
       3208, 783,  2341, 291,  783,  3079, 2007, 2483, 1192, 161,  1453, 1095, 1117, 2248, 3181, 76,   1720, 2133, 1473,
       2919, 2207, 1319, 2203, 73,   3175, 149,  2669, 1184, 78,   1425, 3245, 459,  2875, 2723, 2460, 2441, 290,  1229,
       2917, 1110, 1755, 2166, 2290, 2672, 1125, 1660, 2308, 3280, 2890, 444,  2351, 3080, 3294, 384,  939,  2904, 350,
       164,  3045, 765,  930,  765,  3171, 1902, 2275, 1542, 2834, 2984, 2238, 1345, 1017, 2336, 3312, 315,  121,  2741,
       596,  1466, 2010, 1450, 1123, 2653, 669,  1394, 2237, 1602, 1737, 1030, 3177, 1855, 938,  2633, 2432, 813,  988,
       3260, 1006, 2346, 2256, 2527, 2074, 2528, 2326, 2845, 621,  93,   2632, 3219, 3242, 30,   1014, 3000, 134,  957,
       3074, 690,  1577, 348,  2914, 2021, 2521, 1002, 2587, 435,  1234, 1934, 908,  3130, 1567, 2928, 1403, 2750, 4,
       438,  2566, 1894, 1922, 1054, 716,  1884, 1429, 477,  2871, 1540, 3236, 1546, 3139, 1234, 2660, 1664, 1080, 148,
       584,  660,  1534, 1255, 1154, 1822, 779,  1383, 3005, 1565, 1148, 2209, 2481, 3229, 1492, 3069, 1243, 1915, 2720,
       514,  3170, 2677, 1791, 2469, 245,  1858, 485,  1687},
      {2990, 1725, 2831, 1396, 20,   2939, 731,  3234, 900,  1884, 1486, 590,  3229, 3107, 963,  1187, 1210, 2354, 54,
       1200, 387,  158,  953,  447,  713,  186,  2150, 196,  2482, 403,  675,  2550, 831,  217,  1474, 8,    1604, 2180,
       2190, 2993, 124,  1036, 2641, 878,  1919, 1012, 1375, 560,  309,  3222, 2412, 493,  2680, 1501, 237,  2448, 1720,
       1083, 1921, 2222, 2406, 98,   1449, 1117, 2928, 277,  1468, 121,  1507, 2880, 1275, 1966, 860,  651,  1268, 428,
       2348, 334,  420,  463,  3247, 1584, 2598, 277,  1412, 1489, 3116, 1811, 2951, 2041, 20,   550,  78,   2143, 48,
       2747, 1753, 2437, 1664, 2700, 3049, 392,  2939, 1701, 1922, 1974, 1170, 220,  921,  893,  1882, 1914, 1255, 717,
       1230, 2429, 1155, 1861, 1915, 1862, 1837, 1783, 1369, 2220, 2700, 1989, 2322, 2881, 1433, 573,  869,  1492, 417,
       1412, 2048, 682,  858,  3129, 2484, 3159, 1746, 275,  1985, 3222, 399,  2829, 2364, 203,  1820, 3002, 2434, 1692,
       2718, 1927, 1346, 218,  3006, 58,   3124, 2719, 843,  85,   1970, 1538, 3248, 956,  2460, 236,  1848, 1649, 1788,
       1403, 302,  3106, 999,  2862, 1172, 243,  1256, 217,  2721, 1512, 2765, 518,  1739, 3135, 2965, 797,  1362, 2232,
       2707, 1511, 1923, 2469, 768,  2045, 922,  1229, 1878, 465,  2038, 2841, 493,  2088, 1673, 1768, 581,  73,   1668,
       2021, 1242, 640,  1105, 2028, 1213, 1167, 943,  151,  1269, 2381, 2472, 1084, 2513, 1706, 2398, 830,  2015, 1961,
       2636, 1164, 1803, 663,  308,  906,  1290, 2551, 200,  982,  2757, 497,  1401, 2820, 3147, 2570, 1074, 122,  757,
       1748, 306,  3159, 1253, 2784, 2912, 2418, 2430, 30},
      {2085, 705,  21,   24,   230,  2191, 415,  98,   2194, 3003, 1751, 2306, 900,  2919, 2696, 2655, 242,  711,  239,
       809,  664,  2283, 3223, 2698, 2318, 1226, 2840, 2149, 1301, 113,  2724, 261,  1047, 1932, 772,  1970, 2996, 2684,
       1271, 1274, 290,  1680, 350,  1555, 717,  2547, 1617, 952,  2765, 1944, 2039, 2575, 2756, 1990, 2866, 1113, 1705,
       634,  899,  2925, 2482, 154,  3156, 1135, 1577, 737,  1052, 105,  2959, 1066, 2545, 1617, 686,  1747, 2537, 3103,
       1257, 3210, 2347, 2757, 2150, 2416, 544,  2416, 2470, 1613, 559,  711,  243,  301,  2184, 747,  2368, 3249, 2783,
       1045, 1167, 42,   3041, 793,  1182, 1786, 1024, 1847, 988,  92,   2301, 796,  790,  1408, 1337, 1768, 2432, 2154,
       183,  2725, 2384, 580,  3169, 337,  3312, 2020, 3025, 865,  2328, 3274, 88,   3075, 2364, 1890, 3116, 3271, 1448,
       543,  2198, 1891, 399,  1370, 1079, 713,  870,  26,   142,  2676, 71,   950,  830,  2226, 338,  2938, 2687, 1341,
       1918, 732,  2351, 1092, 2051, 2250, 2584, 2461, 1153, 2792, 2217, 3228, 707,  2720, 2987, 2207, 2962, 2767, 767,
       457,  1156, 2344, 2633, 1318, 3071, 1725, 3302, 626,  1366, 70,   1103, 720,  2273, 2505, 1615, 225,  2353, 294,
       105,  3073, 3269, 635,  678,  2337, 309,  852,  2863, 794,  2899, 1159, 2972, 2180, 1044, 1495, 2074, 900,  1755,
       3035, 684,  3304, 315,  1023, 1344, 2127, 3073, 2067, 1305, 820,  1245, 1585, 2519, 2144, 1052, 1043, 1903, 1754,
       1041, 2352, 1978, 1890, 142,  2237, 2291, 1554, 3321, 1786, 296,  407,  2196, 150,  2660, 2219, 1592, 17,   608,
       133,  1612, 1615, 939,  1353, 1098, 2795, 1537, 1196},
    },
    {
      {3088, 3158, 3291, 1438, 515,  1530, 3206, 117,  1355, 2167, 1524, 427,  3276, 91,   2039, 2544, 1899, 124,  3179,
       3142, 1361, 2295, 712,  1448, 1937, 1310, 3189, 496,  1394, 392,  2470, 1391, 1723, 3262, 3138, 2192, 2738, 1823,
       1091, 607,  1107, 562,  2407, 1552, 866,  349,  2146, 169,  882,  620,  2424, 961,  1401, 387,  2994, 393,  2674,
       1314, 613,  1089, 2434, 3092, 963,  2549, 3271, 709,  182,  1555, 414,  374,  3017, 2516, 3203, 2705, 2642, 1274,
       578,  2959, 1784, 1264, 1926, 1312, 916,  547,  2028, 1500, 2240, 2496, 466,  882,  2785, 3259, 308,  1599, 2127,
       2818, 2813, 1076, 882,  1360, 1120, 2428, 868,  1314, 3053, 2079, 1566, 1530, 2937, 1300, 788,  845,  2459, 394,
       3180, 870,  2133, 712,  432,  1694, 2374, 16,   390,  2104, 2432, 2849, 458,  1007, 1384, 681,  2081, 2217, 272,
       2704, 577,  1462, 333,  424,  2499, 1210, 2834, 2877, 2955, 794,  2064, 2258, 439,  1671, 3163, 2453, 179,  209,
       2522, 1297, 246,  2894, 27,   163,  2438, 2401, 703,  885,  1980, 1221, 1573, 933,  1284, 1733, 640,  3037, 1919,
       1825, 523,  1427, 718,  676,  263,  1885, 865,  2007, 839,  1873, 2517, 733,  2598, 1961, 47,   1265, 1429, 1964,
       1714, 392,  1666, 538,  2525, 298,  565,  1457, 255,  1555, 1355, 1727, 2423, 1311, 2131, 1181, 382,  928,  1952,
       2588, 1757, 2476, 1634, 2929, 174,  1466, 801,  596,  2987, 681,  181,  1241, 2735, 904,  93,   1362, 2274, 3224,
       2295, 1894, 2089, 1757, 1043, 832,  1840, 2892, 413,  2575, 962,  355,  461,  1484, 3293, 744,  398,  2318, 211,
       1426, 607,  1463, 1706, 2807, 2038, 2302, 1364, 2576},
      {403,  3114, 750,  888,  2965, 1781, 356,  993,  126,  235,  1568, 804,  385,  2547, 1414, 1406, 2411, 679,  1839,
       935,  137,  2390, 1856, 2206, 777,  3196, 1132, 504,  2426, 1071, 388,  799,  430,  1427, 3294, 1828, 2382, 2608,
       833,  2900, 321,  1686, 924,  2332, 1409, 1974, 345,  1881, 871,  888,  325,  1470, 2965, 3196, 632,  3049, 1509,
       183,  816,  2325, 2654, 3286, 3032, 1994, 1031, 2282, 1890, 1898, 3254, 1567, 1589, 1977, 1127, 554,  357,  2346,
       3272, 2749, 1413, 1384, 1688, 905,  3279, 1286, 1125, 2769, 1600, 2457, 1608, 284,  649,  996,  2241, 1010, 3305,
       1222, 3087, 735,  1314, 2720, 1980, 1036, 981,  392,  492,  2053, 511,  1056, 465,  791,  2203, 1950, 1635, 1313,
       450,  3252, 2031, 2873, 54,   934,  1528, 1889, 2152, 388,  2032, 1410, 1666, 1250, 1913, 2154, 2294, 955,  372,
       1845, 2936, 1558, 971,  2970, 460,  1100, 736,  2485, 2879, 1524, 2141, 436,  2957, 587,  480,  3021, 2028, 974,
       870,  2533, 204,  1444, 209,  849,  2915, 1049, 2029, 615,  963,  2465, 2063, 1645, 3164, 2100, 2831, 2567, 1444,
       1381, 2723, 3041, 912,  3154, 235,  1978, 942,  1358, 921,  1579, 3137, 184,  2154, 2499, 2735, 722,  1715, 3165,
       3259, 2318, 1234, 2903, 1480, 1075, 1031, 2150, 350,  968,  2666, 3016, 2224, 1576, 1699, 3261, 1244, 933,  2383,
       2931, 2598, 2589, 2986, 3160, 125,  850,  1183, 816,  3306, 1618, 449,  3062, 2419, 2710, 3313, 846,  2970, 1299,
       1913, 1696, 1692, 1043, 1837, 2901, 2690, 51,   2858, 1918, 1822, 1971, 821,  1020, 517,  493,  1069, 1200, 217,
       1495, 2349, 1857, 956,  3034, 90,   2786, 1308, 2194},
      {1770, 2664, 3300, 3193, 1916, 348,  2171, 559,  2953, 375,  1952, 495,  666,  2514, 1484, 102,  2907, 544,  3053,
       923,  788,  1696, 2664, 3094, 215,  2133, 2153, 1530, 1227, 2853, 75,   2699, 2229, 2542, 1149, 2230, 915,  3040,
       1960, 1599, 2510, 3223, 337,  2958, 1867, 2317, 3029, 3229, 1452, 2211, 2226, 602,  423,  428,  2671, 911,  758,
       1288, 1119, 2144, 744,  2241, 2490, 1002, 969,  189,  2824, 1830, 1822, 2955, 2445, 3194, 941,  744,  3023, 2057,
       1400, 2131, 3072, 3285, 1289, 2923, 1237, 1651, 2353, 2899, 219,  393,  682,  2060, 1075, 2333, 267,  355,  273,
       360,  400,  2687, 2795, 2194, 663,  3269, 2455, 3178, 2949, 244,  1534, 1341, 926,  2651, 2771, 284,  1766, 3320,
       2551, 2674, 2028, 2402, 2496, 127,  3316, 2907, 996,  2534, 1464, 2668, 1701, 1692, 1616, 3150, 1879, 2836, 968,
       1562, 805,  2990, 291,  3252, 3028, 2010, 1046, 393,  2478, 890,  3322, 721,  1019, 1669, 1710, 1500, 79,   676,
       2957, 2761, 2810, 1148, 2891, 841,  2500, 3226, 2013, 3314, 2079, 2078, 3103, 3071, 2441, 300,  2831, 3061, 160,
       2950, 193,  2241, 2678, 1697, 386,  688,  2128, 2585, 2586, 2217, 957,  1740, 1482, 2541, 2993, 2589, 594,  942,
       551,  904,  3221, 9,    3252, 340,  47,   2381, 2011, 2657, 1486, 1800, 1097, 1414, 2918, 2750, 2447, 2687, 859,
       2705, 1688, 1597, 2755, 113,  691,  2416, 1068, 1891, 1541, 2697, 675,  745,  1381, 612,  500,  921,  1555, 1151,
       2495, 3203, 1234, 3254, 2314, 450,  2269, 1588, 2969, 1767, 1962, 3202, 435,  1679, 1767, 2847, 1986, 2934, 3209,
       134,  2180, 579,  1729, 2679, 3158, 756,  110,  870},
      {1482, 475,  3327, 3108, 681,  2986, 56,   1638, 315,  2872, 2581, 3196, 3252, 653,  32,   240,  2167, 2240, 1006,
       1667, 2691, 1440, 1732, 649,  989,  899,  501,  2558, 3080, 1759, 2552, 985,  1265, 561,  1847, 2445, 2781, 611,
       1813, 588,  54,   93,   2308, 2873, 1584, 746,  1655, 2726, 2583, 1245, 913,  2675, 2606, 3190, 2787, 3058, 1679,
       1410, 2061, 2458, 2795, 703,  302,  2183, 872,  891,  1367, 1619, 2836, 2654, 1735, 148,  635,  2871, 1014, 620,
       3266, 1764, 1496, 1112, 3008, 653,  2679, 718,  1948, 1368, 437,  2699, 614,  321,  2259, 2059, 825,  842,  1224,
       2911, 3078, 680,  641,  722,  1522, 2980, 795,  2746, 2059, 1203, 2842, 2255, 319,  140,  1386, 271,  2022, 693,
       2169, 2305, 342,  3166, 1149, 2067, 2910, 840,  2244, 1734, 723,  1085, 1912, 3003, 1645, 3323, 3142, 628,  2435,
       744,  1882, 1049, 2683, 1920, 1827, 1837, 918,  2920, 1755, 2510, 1909, 845,  2436, 1139, 1086, 1295, 2336, 4,
       545,  625,  1732, 1359, 2377, 2446, 3056, 587,  1335, 2029, 2114, 2820, 529,  3222, 2444, 953,  1577, 3238, 3093,
       434,  405,  3211, 2502, 436,  3299, 2842, 3004, 1373, 703,  1092, 900,  2532, 2060, 1676, 2409, 1049, 369,  2596,
       1976, 2096, 1229, 1792, 2769, 818,  2579, 153,  2417, 953,  3020, 1325, 2587, 2845, 1570, 3242, 3048, 413,  2967,
       3248, 1443, 279,  1686, 305,  1935, 1490, 760,  778,  3125, 1543, 268,  3011, 2728, 1056, 1124, 723,  1430, 598,
       1015, 3264, 1627, 140,  1967, 3125, 952,  699,  1963, 29,   881,  2409, 970,  77,   1481, 57,   2914, 2297, 2335,
       2405, 1218, 2780, 1720, 1332, 2546, 1745, 2855, 2342},
    },
    {
      {885,  3111, 448,  1384, 1374, 1355, 584,  2718, 1733, 403,  2790, 1326, 2696, 285,  1010, 2601, 2261, 556,  579,
       2474, 692,  2673, 1500, 1622, 2893, 2352, 2084, 1723, 1802, 2972, 2110, 2884, 855,  130,  801,  3021, 346,  786,
       222,  1208, 1273, 1722, 991,  3258, 534,  920,  638,  334,  1092, 560,  728,  1994, 2960, 2993, 1807, 2447, 1162,
       3294, 953,  2096, 1398, 1073, 851,  2255, 1132, 175,  1005, 2126, 1127, 1567, 1141, 1073, 2702, 1186, 106,  1480,
       2410, 1877, 2359, 2194, 386,  38,   2341, 352,  2500, 2085, 374,  3285, 1509, 2754, 3206, 898,  562,  435,  516,
       559,  1204, 2296, 717,  1772, 516,  1453, 471,  1376, 501,  1222, 2489, 639,  1925, 219,  544,  1590, 1827, 986,
       1275, 671,  2796, 2732, 2761, 1516, 1101, 883,  2776, 1017, 2407, 2515, 1922, 2056, 267,  1449, 2148, 3114, 1140,
       2858, 385,  1755, 997,  1143, 678,  1505, 2010, 1125, 622,  458,  1908, 277,  2373, 899,  2523, 1706, 2388, 2990,
       2961, 1534, 49,   2682, 3301, 372,  1793, 377,  1121, 549,  1148, 2459, 692,  1671, 254,  2924, 109,  1657, 2497,
       2174, 3073, 2818, 2538, 849,  2795, 745,  1192, 516,  672,  2497, 3137, 141,  3019, 503,  2003, 660,  1130, 217,
       2840, 2926, 2822, 1134, 1663, 2556, 2006, 1484, 1938, 1020, 2540, 1953, 1461, 2652, 511,  1097, 2391, 700,  3225,
       2933, 3054, 781,  1194, 2367, 72,   741,  3136, 1882, 3118, 2210, 1432, 2616, 1851, 794,  2230, 1030, 644,  802,
       2097, 254,  67,   1200, 2881, 2444, 969,  590,  1990, 1984, 2545, 2450, 1190, 394,  2933, 2577, 72,   1144, 4,
       972,  2699, 2328, 2872, 946,  1457, 1566, 392,  1904},
      {2457, 1985, 3134, 596,  610,  1387, 1668, 3213, 1930, 784,  1113, 2459, 2191, 264,  2581, 1789, 2878, 1774, 902,
       1485, 464,  1514, 2677, 2361, 2847, 2819, 1680, 913,  2644, 2139, 2455, 2998, 810,  1090, 3015, 1345, 3256, 2113,
       2550, 2684, 2049, 1484, 2106, 1048, 1758, 3075, 736,  1468, 193,  1827, 1001, 750,  349,  1003, 1118, 2405, 678,
       1676, 112,  52,   398,  3283, 1573, 2336, 574,  3213, 2445, 736,  992,  2990, 2718, 1773, 2525, 2523, 2672, 1566,
       2853, 1772, 20,   380,  1840, 813,  1945, 892,  926,  911,  2802, 792,  3037, 2021, 2383, 882,  2759, 2671, 2928,
       1045, 1650, 358,  2632, 1516, 1315, 935,  2820, 2395, 2839, 21,   1744, 2346, 1212, 3163, 1769, 1463, 2554, 864,
       3204, 3028, 2005, 2222, 1189, 3131, 1814, 1384, 644,  159,  2855, 2824, 546,  1918, 2493, 900,  1588, 813,  2384,
       738,  3097, 3028, 1759, 1236, 2693, 2030, 2944, 1529, 2816, 2425, 2472, 2984, 1710, 2209, 1701, 1684, 781,  3188,
       2469, 2795, 123,  846,  2546, 638,  519,  654,  2944, 1332, 1381, 2563, 767,  2088, 3054, 1611, 1075, 2453, 2164,
       3087, 1177, 1462, 3197, 876,  2474, 3020, 412,  133,  2470, 497,  1007, 919,  2283, 2676, 1184, 2499, 1319, 567,
       91,   850,  1189, 1308, 1338, 2563, 2379, 2340, 435,  3184, 1273, 31,   1337, 1272, 1565, 1124, 2587, 1192, 2128,
       1567, 137,  124,  1937, 2157, 570,  2580, 1646, 528,  1993, 290,  3069, 1002, 3047, 2119, 997,  1879, 2757, 939,
       2855, 1856, 370,  1608, 1867, 3061, 73,   2388, 1753, 66,   159,  1717, 739,  2778, 1072, 2515, 2230, 52,   1278,
       2600, 2810, 1476, 1310, 3087, 781,  3052, 562,  1643},
      {663,  880,  1210, 2183, 730,  3275, 955,  721,  1365, 2179, 327,  1753, 2892, 102,  2790, 1321, 1932, 1873, 244,
       1634, 1522, 298,  324,  48,   1716, 2502, 1270, 2467, 2082, 3206, 1663, 1111, 1572, 1393, 151,  1651, 2828, 2285,
       1237, 3082, 2381, 725,  2731, 296,  431,  1340, 1721, 1255, 1216, 2036, 668,  2254, 3310, 1390, 1163, 2411, 503,
       354,  1104, 1988, 3189, 2650, 1588, 2739, 1673, 932,  1701, 1717, 1661, 3070, 907,  1039, 2388, 2779, 2959, 2610,
       299,  1465, 6,    2756, 1105, 25,   1557, 465,  364,  3066, 2452, 2058, 2823, 2563, 654,  503,  687,  941,  907,
       3269, 1310, 1749, 939,  1240, 1873, 2995, 1490, 1262, 113,  1198, 733,  3160, 1137, 1456, 2309, 554,  1875, 1507,
       2121, 1310, 2947, 1653, 3083, 2557, 234,  1130, 2753, 787,  2883, 318,  723,  1629, 1284, 2683, 424,  1162, 1203,
       3079, 187,  1902, 2054, 2459, 2157, 490,  856,  215,  627,  2474, 2240, 116,  1292, 2918, 777,  1258, 584,  1082,
       1362, 475,  1464, 1550, 2312, 2787, 668,  3079, 769,  2995, 131,  1004, 2108, 660,  252,  388,  154,  1438, 2036,
       2241, 3024, 1320, 1245, 2217, 2892, 604,  3277, 2899, 204,  3249, 2676, 966,  956,  1291, 967,  1369, 139,  2174,
       1727, 2947, 2813, 3061, 717,  887,  2608, 2631, 1477, 1098, 1801, 1230, 1791, 881,  1218, 1525, 1569, 1262, 16,
       2338, 2208, 1140, 1457, 786,  2188, 198,  429,  3307, 1666, 556,  1484, 542,  62,   647,  457,  2761, 269,  2017,
       851,  480,  19,   2211, 53,   1595, 1717, 1541, 352,  3320, 653,  1875, 449,  2813, 1545, 131,  107,  1381, 454,
       1487, 2819, 518,  1229, 1646, 1595, 874,  2079, 2796},
      {2280, 1085, 482,  3287, 2431, 2774, 1707, 705,  2893, 644,  2400, 452,  713,  2813, 1523, 2000, 1126, 2142, 3204,
       1146, 2179, 1775, 3190, 618,  2959, 1617, 273,  2795, 321,  2249, 1459, 1474, 718,  2875, 434,  147,  935,  385,
       2980, 864,  2204, 2973, 1694, 2581, 684,  617,  581,  3082, 3158, 28,   729,  880,  1509, 585,  2454, 282,  953,
       1414, 1168, 1689, 347,  39,   2313, 3238, 1987, 1228, 137,  2766, 230,  1122, 435,  489,  1344, 2743, 1707, 2755,
       1736, 2935, 2141, 2190, 1349, 1407, 1427, 2006, 1912, 1607, 622,  1262, 657,  477,  55,   2995, 442,  1026, 1153,
       1753, 311,  95,   37,   209,  769,  1358, 2975, 2667, 2041, 516,  2558, 2084, 947,  1635, 567,  836,  2612, 797,
       133,  1706, 1601, 3228, 2849, 2236, 1892, 479,  1184, 2344, 677,  546,  2481, 1272, 2277, 1555, 765,  714,  2858,
       336,  1361, 1076, 1670, 1320, 873,  874,  2601, 1797, 2644, 1300, 1226, 461,  2519, 3202, 1056, 426,  1563, 2852,
       1586, 1159, 2467, 577,  2021, 1895, 1321, 2218, 971,  2101, 289,  1994, 2247, 1657, 3272, 3213, 1969, 1076, 782,
       1494, 388,  1402, 1540, 86,   1671, 822,  2467, 2269, 2013, 2503, 397,  1190, 1979, 1648, 3307, 1223, 1032, 1381,
       1159, 1623, 3073, 3318, 453,  1629, 1988, 2445, 2242, 967,  3048, 2533, 2865, 1926, 129,  2832, 1299, 1213, 1596,
       446,  331,  831,  3097, 2158, 2440, 2515, 1565, 2265, 2035, 2105, 215,  633,  508,  3320, 1807, 1165, 772,  576,
       1173, 354,  3067, 1560, 3011, 1782, 664,  21,   3011, 1718, 1535, 1372, 2561, 73,   3225, 613,  2058, 517,  1299,
       571,  2447, 918,  876,  2846, 533,  86,   283,  870},
    },
  };

  const uint16_t reference_vec[4][256] = {
    {1389, 98,   1885, 1661, 1426, 1440, 1669, 82,   770,  192,  1258, 1489, 1836, 397,  914,  2246, 3322, 1401, 2261,
     1351, 2381, 129,  1386, 2546, 3233, 2248, 3117, 285,  809,  1777, 1397, 1041, 1929, 2350, 2779, 234,  230,  2248,
     426,  2381, 2009, 1493, 2419, 3009, 2348, 1265, 2107, 446,  1580, 1545, 668,  1304, 534,  912,  1351, 845,  1988,
     2193, 1635, 550,  2467, 3161, 2765, 1711, 43,   2262, 3174, 1409, 3259, 856,  547,  2849, 1591, 783,  1234, 452,
     2323, 2023, 1547, 836,  1105, 1512, 1336, 1822, 3008, 743,  2658, 2201, 2834, 2126, 144,  1963, 2519, 1447, 1794,
     1769, 940,  2593, 428,  463,  1016, 579,  3179, 1753, 3298, 1892, 777,  1362, 1775, 1497, 2883, 1950, 2159, 1538,
     1273, 2553, 2203, 820,  1338, 338,  441,  982,  2851, 2339, 1384, 152,  947,  727,  2506, 139,  3192, 3108, 1558,
     1907, 3090, 1700, 1545, 707,  968,  345,  2962, 495,  612,  959,  309,  3202, 2925, 150,  696,  1673, 2233, 2161,
     2477, 1949, 3232, 3038, 2108, 2422, 317,  3136, 1552, 965,  585,  2010, 1832, 1791, 2950, 2949, 1389, 327,  1191,
     3216, 3015, 1340, 1628, 1233, 3063, 2705, 1672, 2154, 2017, 509,  902,  1000, 3112, 216,  2182, 2370, 2672, 3302,
     1393, 2578, 142,  1947, 1685, 1370, 224,  2131, 1591, 774,  2374, 2693, 1499, 396,  3227, 3305, 201,  1791, 841,
     432,  923,  1256, 3297, 3071, 973,  3134, 849,  3032, 2471, 2783, 1479, 2360, 2746, 2,    2204, 2068, 1217, 2768,
     2459, 533,  1701, 1302, 1880, 2925, 616,  1899, 1850, 582,  2791, 1717, 2835, 1844, 2821, 3025, 1322, 2264, 657,
     2352, 2257, 1357, 748,  2260, 1708, 586,  1919, 657},
    {2974, 886,  937,  2558, 1738, 2323, 1216, 2716, 1204, 1733, 1855, 1549, 2796, 1115, 267,  2863, 3143, 69,   2148,
     2713, 3004, 1023, 2348, 2733, 576,  1850, 872,  2940, 805,  1150, 51,   1484, 1765, 1616, 2678, 907,  2006, 1717,
     2854, 2718, 2098, 1562, 705,  158,  2018, 1160, 2354, 1113, 2847, 319,  432,  1851, 3308, 2709, 1229, 1152, 3147,
     554,  2027, 2818, 2467, 2326, 1021, 869,  144,  1877, 558,  453,  981,  2011, 913,  3201, 2123, 1696, 1228, 3084,
     1835, 1228, 3071, 1868, 1304, 205,  237,  3049, 610,  742,  2476, 3024, 574,  1984, 243,  785,  2534, 1133, 2514,
     586,  2719, 80,   1273, 3173, 65,   2899, 1825, 1708, 2409, 2181, 1068, 1962, 1646, 372,  1409, 1836, 2914, 755,
     3209, 3213, 1382, 1625, 3143, 220,  536,  1602, 3032, 2144, 504,  949,  1802, 1631, 595,  387,  1333, 1882, 499,
     548,  3256, 3181, 2217, 3304, 1820, 1061, 315,  2372, 1113, 912,  2471, 1741, 266,  940,  2982, 1537, 1888, 1246,
     2304, 1300, 913,  918,  912,  596,  513,  680,  1378, 900,  1580, 2792, 3211, 2853, 380,  1496, 2561, 2161, 1597,
     2747, 2470, 1376, 3201, 1225, 2996, 3040, 1471, 2754, 460,  661,  538,  2751, 3247, 3145, 1808, 2885, 1579, 3305,
     1111, 633,  929,  813,  898,  788,  2618, 2134, 3257, 2359, 1424, 1966, 689,  677,  874,  1477, 769,  1457, 1325,
     671,  1185, 419,  1147, 1493, 1396, 3018, 803,  2234, 2821, 1982, 22,   3324, 437,  2751, 1957, 2865, 978,  1975,
     3222, 653,  1062, 2780, 965,  1191, 2923, 1853, 3225, 2085, 387,  793,  3075, 2528, 1442, 1809, 206,  247,  697,
     207,  3103, 1978, 870,  1899, 1589, 148,  66,   2522},
    {2797, 2548, 495,  2301, 2259, 2863, 1817, 3199, 2961, 1869, 1967, 1848, 1369, 627,  1737, 421,  2606, 1669, 1498,
     2372, 336,  223,  2031, 741,  85,   1334, 3088, 3116, 89,   1391, 1373, 1653, 2711, 2607, 2836, 1711, 357,  2489,
     3325, 1217, 1602, 3295, 2057, 1395, 860,  2925, 2923, 1910, 822,  1554, 1891, 39,   1609, 2989, 1245, 1509, 2434,
     281,  159,  90,   1613, 1982, 2579, 1994, 743,  852,  138,  1976, 3268, 656,  2938, 302,  1239, 2847, 1542, 31,
     3174, 283,  252,  2070, 238,  923,  1340, 1515, 1089, 2705, 3217, 2273, 244,  1553, 179,  376,  1701, 829,  3212,
     2287, 1251, 2909, 3089, 2103, 1363, 366,  610,  1727, 2452, 2681, 185,  344,  2674, 2369, 1847, 2588, 467,  1580,
     1753, 3239, 2811, 3069, 2193, 2986, 1877, 2632, 2928, 415,  1928, 2542, 892,  169,  549,  1323, 3165, 283,  2320,
     1690, 345,  2810, 2301, 830,  460,  1034, 2336, 1549, 97,   2181, 1516, 3144, 2463, 2434, 1915, 3264, 1814, 310,
     1870, 2868, 2841, 2970, 1834, 2293, 932,  1135, 1741, 384,  2388, 2918, 3131, 2824, 968,  1345, 1454, 3261, 2594,
     2933, 1610, 2797, 1437, 2285, 2392, 1996, 153,  2478, 2249, 395,  1132, 1976, 1686, 2236, 2166, 3301, 3226, 15,
     3121, 1857, 1090, 550,  2643, 3081, 2482, 3155, 2849, 1286, 816,  1870, 938,  438,  2066, 993,  116,  528,  146,
     1574, 1608, 2253, 3314, 1462, 1831, 1985, 1971, 934,  2638, 1534, 288,  786,  253,  1111, 1130, 1640, 2189, 1450,
     1893, 1459, 139,  1736, 1807, 2647, 3206, 2853, 1800, 313,  522,  214,  388,  2788, 726,  830,  1111, 1097, 479,
     1888, 363,  1700, 2465, 2327, 2348, 340,  381,  189},
    {581,  1724, 644,  1978, 820,  994,  1447, 1071, 250,  703,  1427, 1852, 839,  1481, 33,   752,  2434, 1641, 88,
     44,   1367, 3148, 989,  183,  2315, 2078, 3161, 281,  284,  225,  208,  1549, 3256, 3129, 2861, 2673, 284,  2901,
     2806, 930,  2465, 1919, 2592, 2163, 374,  32,   348,  1495, 490,  813,  2438, 2694, 220,  731,  3314, 1726, 2078,
     3184, 309,  231,  2006, 1939, 3241, 2701, 1128, 789,  1207, 2370, 1285, 1049, 721,  2666, 1690, 2799, 2423, 1457,
     2744, 2440, 2226, 3235, 3238, 1658, 205,  2600, 1012, 1510, 205,  707,  1367, 881,  2638, 2573, 1385, 87,   1236,
     262,  1133, 1867, 2963, 3110, 807,  2478, 246,  2932, 2422, 2451, 565,  1867, 88,   315,  117,  1783, 748,  2158,
     1607, 1105, 647,  1765, 1316, 456,  492,  2661, 1066, 384,  864,  2707, 298,  2770, 2506, 444,  2202, 2040, 3250,
     2924, 3217, 802,  2343, 168,  1379, 3220, 2797, 2995, 2794, 3041, 1199, 591,  206,  1193, 1192, 1094, 1692, 677,
     477,  2110, 1739, 1996, 3058, 2438, 464,  1323, 2209, 2786, 469,  1801, 1230, 1022, 2006, 2634, 1652, 2343, 1926,
     1862, 1675, 2995, 3168, 417,  2330, 858,  946,  1866, 1395, 2064, 2657, 2334, 2283, 1369, 1721, 1938, 20,   514,
     57,   1767, 356,  1088, 1775, 1596, 3053, 1638, 1026, 1692, 365,  769,  1374, 2112, 2866, 775,  2759, 2905, 246,
     307,  2162, 1390, 3208, 980,  1309, 1189, 3170, 3215, 1240, 2056, 2162, 3163, 389,  2836, 2929, 36,   3162, 435,
     2437, 1045, 2429, 1273, 1701, 101,  2706, 464,  3145, 802,  2609, 1697, 319,  130,  1530, 3107, 3100, 1013, 1636,
     263,  2662, 291,  3137, 1695, 158,  1169, 3277, 2351},
  };

  const uint16_t reference_result[4][256] = {
    {249,  2860, 2619, 3104, 2858, 1535, 2900, 1510, 872,  3003, 1705, 2684, 383,  2925, 195,  3323, 319,  2901, 1863,
     3163, 767,  1376, 1241, 3208, 1110, 1823, 3165, 1995, 1427, 1497, 2912, 531,  1117, 1546, 2064, 1087, 2986, 2695,
     2370, 2620, 1835, 1805, 900,  1533, 1412, 2712, 2276, 2839, 487,  2593, 2542, 2548, 2403, 3135, 407,  545,  1626,
     3237, 79,   507,  2496, 2826, 2449, 165,  127,  1763, 164,  1688, 1128, 679,  1870, 2582, 2655, 1821, 578,  93,
     1335, 283,  3283, 2873, 2199, 1016, 653,  720,  2497, 390,  1114, 3043, 453,  2726, 2110, 1565, 1082, 1422, 3040,
     684,  1353, 2824, 2870, 1544, 2079, 3261, 283,  1433, 671,  3178, 1698, 2570, 2417, 1794, 658,  1886, 1825, 2390,
     756,  2493, 615,  2758, 2900, 2101, 327,  3108, 2920, 1653, 192,  1625, 3160, 1139, 595,  1278, 86,   1885, 2963,
     2329, 3309, 2600, 2363, 834,  2526, 2009, 2088, 2353, 365,  593,  182,  2666, 802,  2167, 2044, 1191, 975,  838,
     1894, 1559, 771,  1654, 976,  1336, 2378, 2225, 2929, 2862, 2303, 1979, 2152, 901,  16,   478,  3209, 3134, 2715,
     2281, 400,  3327, 871,  642,  2065, 1214, 2912, 1295, 1863, 2392, 2657, 638,  1471, 837,  2334, 30,   2131, 1848,
     2254, 3168, 1866, 1191, 2152, 178,  3098, 299,  2901, 2243, 304,  711,  3090, 3207, 671,  3234, 1963, 2804, 2735,
     3311, 10,   200,  299,  227,  64,   1953, 2806, 3182, 1171, 420,  764,  403,  85,   309,  94,   779,  2178, 3054,
     2116, 1985, 1612, 65,   920,  244,  2712, 1118, 2182, 1594, 317,  1480, 503,  254,  2119, 304,  1619, 510,  3218,
     1738, 2495, 2088, 2015, 999,  1741, 1147, 1273, 542},
    {2486, 3004, 1757, 12,   2606, 2368, 2719, 2440, 1351, 1197, 1393, 3102, 604,  1534, 290,  429,  2488, 2114, 470,
     2019, 991,  1771, 3065, 3319, 62,   2879, 2445, 2630, 2116, 2256, 2069, 777,  2826, 2924, 1563, 2637, 487,  843,
     345,  2002, 2069, 1430, 3092, 1556, 2791, 2202, 3168, 2965, 2405, 157,  1545, 2572, 2070, 1382, 1695, 119,  2785,
     1507, 1788, 30,   3296, 2396, 493,  2308, 2180, 1235, 1238, 2463, 1878, 1843, 1435, 1097, 1929, 2459, 2632, 2540,
     326,  1701, 477,  775,  403,  2758, 715,  1549, 1450, 1726, 719,  2986, 1901, 2960, 1222, 1732, 1408, 2421, 1049,
     695,  1916, 2192, 1043, 575,  1430, 2164, 1178, 370,  1384, 716,  2075, 3049, 2849, 2273, 283,  481,  1401, 1472,
     1092, 642,  2556, 344,  751,  2586, 230,  1286, 2842, 2406, 2094, 1955, 1073, 738,  2221, 3059, 3013, 1939, 1595,
     3225, 730,  1022, 2159, 2305, 2789, 2377, 2050, 2209, 2433, 1652, 2812, 2839, 68,   2813, 466,  1459, 1118, 1841,
     355,  3084, 1217, 2114, 1874, 2559, 663,  1799, 1310, 2323, 843,  1568, 727,  1145, 3263, 1175, 1584, 1991, 1930,
     1550, 1359, 1286, 1845, 3033, 1843, 2681, 1454, 2443, 1980, 869,  2290, 502,  1307, 1228, 1552, 2672, 919,  3281,
     1405, 1194, 3277, 2627, 170,  577,  2970, 2597, 1678, 2170, 443,  1079, 2745, 2548, 2941, 2140, 3052, 656,  236,
     808,  994,  2298, 1569, 1657, 467,  507,  2256, 2613, 1549, 2594, 2571, 3248, 2304, 940,  771,  2129, 3053, 2069,
     433,  2792, 658,  708,  2714, 535,  2499, 2434, 1778, 1411, 631,  2527, 3301, 2980, 2050, 3311, 2906, 186,  2621,
     2856, 2482, 692,  2502, 2018, 1483, 2664, 2323, 1656},
    {2048, 1915, 800,  1977, 2361, 1106, 996,  1964, 242,  522,  159,  583,  1954, 1765, 1546, 1013, 1528, 137,  1195,
     1578, 1608, 1172, 1063, 2826, 627,  299,  175,  1052, 2828, 1137, 291,  2662, 240,  559,  1059, 1589, 980,  1913,
     482,  558,  1186, 2825, 903,  235,  2597, 176,  316,  1438, 2369, 1979, 2860, 1894, 822,  2372, 1020, 3320, 393,
     211,  1407, 2103, 67,   2056, 1844, 1477, 2493, 1887, 1584, 156,  1648, 2238, 1435, 1513, 2456, 1219, 84,   2203,
     875,  731,  575,  3177, 2814, 1160, 2206, 2176, 1032, 2574, 2842, 1981, 2854, 2326, 1757, 2206, 1503, 2456, 186,
     898,  1164, 3229, 752,  19,   899,  1463, 2638, 439,  1912, 2552, 2452, 2682, 1677, 566,  3126, 2382, 2500, 1985,
     2569, 732,  2393, 419,  693,  2883, 692,  2386, 2798, 2456, 2039, 2513, 1978, 2961, 1367, 2875, 2834, 1338, 1703,
     1054, 1003, 2849, 1486, 1961, 365,  2977, 2999, 1168, 966,  1311, 3061, 2203, 1665, 381,  3020, 919,  1062, 356,
     3127, 739,  2581, 744,  973,  1861, 267,  132,  187,  451,  734,  2161, 2134, 3282, 2888, 950,  2620, 1809, 697,
     2383, 2110, 1871, 3079, 464,  1867, 1307, 1596, 987,  3279, 2595, 2483, 1270, 2806, 583,  3196, 1158, 2070, 2001,
     2237, 2569, 251,  3119, 922,  2995, 2907, 1565, 2736, 2649, 1978, 2013, 842,  2454, 1423, 2020, 27,   2256, 2218,
     2431, 2891, 1170, 1045, 1851, 170,  2064, 1013, 2986, 3104, 2196, 1997, 1357, 1892, 2126, 1058, 946,  130,  3059,
     3273, 1585, 443,  2779, 2992, 805,  2556, 150,  3190, 1174, 355,  1437, 1123, 1588, 1956, 124,  2365, 2339, 1121,
     1711, 1961, 595,  2459, 513,  3315, 76,   1329, 3220},
    {364,  2406, 2178, 1834, 1593, 2500, 1935, 433,  2867, 1743, 1784, 79,   1056, 1338, 114,  1097, 719,  1911, 3011,
     1198, 753,  1936, 1230, 2897, 3179, 1166, 2553, 2903, 2317, 482,  1908, 3135, 2688, 2493, 1687, 1562, 62,   840,
     2858, 2066, 2053, 451,  1909, 445,  1081, 174,  1693, 1685, 1593, 31,   1281, 1451, 2585, 1935, 3177, 1817, 1150,
     2475, 2163, 3253, 1240, 2885, 645,  2562, 2150, 1812, 2570, 2241, 2650, 3212, 131,  2605, 665,  2856, 1225, 403,
     2657, 2245, 2453, 1745, 2524, 1193, 2015, 136,  2122, 1174, 2790, 248,  2777, 1079, 665,  264,  997,  1428, 963,
     2370, 2902, 802,  3208, 1208, 1953, 1899, 2436, 1816, 686,  3177, 1303, 1011, 1594, 1420, 671,  1674, 1462, 2645,
     659,  2622, 700,  526,  2569, 12,   2612, 1450, 1995, 3179, 13,   1268, 149,  168,  1068, 2157, 362,  2082, 750,
     1251, 78,   1138, 3192, 64,   1927, 3035, 390,  699,  189,  518,  549,  2950, 637,  3082, 3056, 2577, 683,  620,
     1097, 2793, 2598, 2374, 3188, 1208, 3098, 3045, 1896, 1974, 510,  981,  628,  2021, 2350, 929,  32,   2514, 2161,
     1758, 3255, 1874, 2705, 315,  943,  217,  825,  1414, 1082, 1556, 3201, 2351, 946,  278,  2695, 1419, 1692, 692,
     2120, 1315, 159,  211,  2431, 2737, 2586, 979,  1939, 1986, 3191, 1761, 899,  2986, 269,  1086, 1831, 2293, 1346,
     1337, 1175, 3175, 1871, 1893, 674,  836,  1065, 222,  2838, 3316, 862,  1545, 3111, 2894, 991,  1978, 43,   823,
     864,  3042, 2117, 1862, 1747, 1009, 450,  2687, 2932, 3040, 1242, 874,  3016, 1409, 2025, 2884, 1681, 1460, 1433,
     1216, 876,  1016, 1319, 3219, 1894, 1915, 2634, 1263},
  };

  Zq internal_A[256 * 4 * 4] = {0};
  PolyMatrix<256, 4, 4, Zq> host_A(internal_A);
  for (int row = 0; row < K; row++) {
    for (int col = 0; col < K; col++) {
      for (int i = 0; i < 256; i++) {
        Zq zq_coeff = Zq(reference_matrix[row][col][i]);
        host_A[row][col][i] = zq_coeff;
      }
    }
  }

  Zq internal_x[256 * 4] = {0};
  PolyVec<256, 4, Zq> host_x(internal_x);
  Zq internal_y[256 * 4] = {0};
  PolyVec<256, 4, Zq> host_y(internal_y);
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < 256; j++) {
      host_x[i][j] = Zq(reference_vec[i][j]);
    }
  }

  Zq* d_A;
  Zq* d_x;
  Zq* d_y;
  cudaMalloc(&d_A, PolyMatrix<256, 4, 4, Zq>::byte_size());
  cudaMalloc(&d_x, PolyVec<256, 4, Zq>::byte_size());
  cudaMalloc(&d_y, PolyVec<256, 4, Zq>::byte_size());
  cudaMemcpy(d_A, host_A.data(), PolyMatrix<256, 4, 4, Zq>::byte_size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, host_x.data(), PolyVec<256, 4, Zq>::byte_size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, host_y.data(), PolyVec<256, 4, Zq>::byte_size(), cudaMemcpyHostToDevice);

  test_matrix_vec_mult<4><<<1, 128>>>(d_A, d_x, d_y);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA error: " << cudaGetErrorString(cudaGetLastError());
  cudaMemcpy(host_y.data(), d_y, PolyVec<256, 4, Zq>::byte_size(), cudaMemcpyDeviceToHost);

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < 256; j++) {
      ASSERT_EQ(host_y[i][j].raw(), reference_result[i][j]);
    }
  }
}

template <const uint8_t d>
__global__ void encode_compress_kernel(const Zq* input, uint8_t* output)
{
  Poly<256, Zq> input_poly(const_cast<Zq*>(input));
  // __shared__ uint16_t temp[256];
  byte_encode_compress<d>(input_poly, output);
  // byte_encode(temp, output, d);
}

template <const uint8_t d>
__global__ void decode_decompress_kernel(const uint8_t* input, Zq* output)
{
  Poly<256, Zq> output_poly(const_cast<Zq*>(output));
  // __shared__ uint16_t temp[256];
  byte_decode_decompress<d>(input, output_poly);
  // byte_encode(temp, output, d);
}

// Template for a single test instance
template <uint8_t I>
void run_single_test(
  const uint16_t* reference_input,
  const uint8_t reference_result[12][384],
  const uint16_t reference_decompressed[12][256])
{
  uint8_t host_result[384];
  Zq host_decompressed[256];

  Zq* d_input;
  cudaMalloc(&d_input, Poly<256, Zq>::byte_size());
  cudaMemcpy(d_input, reference_input, Poly<256, Zq>::byte_size(), cudaMemcpyHostToDevice);

  uint8_t* d_result;
  cudaMalloc(&d_result, 384 * sizeof(uint8_t));
  encode_compress_kernel<I><<<1, 32>>>(d_input, d_result);
  cudaMemcpy(host_result, d_result, 384 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  for (uint32_t j = 0; j < 384; ++j) {
    ASSERT_EQ(reference_result[I - 1][j], host_result[j]) << "encode error: i = " << int(I) << " , j = " << j;
  }

  Zq* d_decompressed;
  cudaMalloc(&d_decompressed, 256 * sizeof(Zq));
  decode_decompress_kernel<I><<<1, 32>>>(d_result, d_decompressed);
  cudaMemcpy(host_decompressed, d_decompressed, 256 * sizeof(Zq), cudaMemcpyDeviceToHost);

  for (uint32_t j = 0; j < 256; ++j) {
    ASSERT_EQ(reference_decompressed[I - 1][j], host_decompressed[j].raw())
      << "decompress error: i = " << int(I) << " , j = " << j;
  }

  cudaFree(d_input);
  cudaFree(d_result);
  cudaFree(d_decompressed);
}

// Compile-time unrolled loop
template <std::size_t... Is>
void run_all_tests(
  const uint16_t* reference_input,
  const uint8_t reference_result[12][384],
  const uint16_t reference_decompressed[12][256],
  std::index_sequence<Is...>)
{
  (run_single_test<Is + 1>(reference_input, reference_result, reference_decompressed), ...);
}

TEST_F(KyberTest, CompressEncode)
{
  const uint16_t reference_input[256] = {
    0xbf,  0x656, 0xb97, 0x72c, 0xc8,  0x103, 0x3b9, 0x906, 0x1c4, 0xc11, 0x42c, 0xad9, 0xbed, 0x2dd, 0xb59, 0x6ad,
    0x58d, 0x5c1, 0x1eb, 0x3ec, 0x64b, 0x764, 0x658, 0x925, 0x39,  0xa0d, 0xbf,  0x5c,  0x4bc, 0x1e9, 0xa6d, 0x23a,
    0x970, 0x5bc, 0x1bf, 0xa7d, 0x2bd, 0x3c1, 0x12a, 0xabc, 0x4b2, 0x16b, 0x57a, 0x99a, 0x2e,  0xa1d, 0x7d6, 0x150,
    0xab7, 0x857, 0xa34, 0x120, 0x1f5, 0x308, 0x579, 0xeb,  0x753, 0x46c, 0x315, 0x2f6, 0xcbf, 0x76f, 0x392, 0x146,
    0x73b, 0x81b, 0x774, 0x631, 0x117, 0xc73, 0x265, 0x8a4, 0x8bf, 0x95f, 0x924, 0x63d, 0x60c, 0x97c, 0xcb3, 0x2fd,
    0x68f, 0x92,  0xa23, 0xa72, 0xa22, 0x7a9, 0x88f, 0x343, 0x247, 0xa7a, 0x30a, 0x2,   0x70f, 0x79b, 0xacd, 0x604,
    0x94c, 0xc34, 0x371, 0xa54, 0x6b7, 0xb06, 0x270, 0x98,  0x62d, 0x462, 0x3bf, 0x26b, 0x131, 0x73,  0x1,   0x72,
    0x9ad, 0x191, 0x60f, 0x478, 0x922, 0xa5e, 0x208, 0x4c,  0x2c9, 0x592, 0xa7a, 0x9ae, 0x2f9, 0xb81, 0xc63, 0x2c2,
    0x8,   0xbdf, 0xafe, 0xbb4, 0x379, 0xc28, 0x94d, 0xbca, 0x241, 0x214, 0x8f4, 0x625, 0x5b,  0x659, 0xcd4, 0x8d0,
    0x974, 0x7ae, 0x3c6, 0x5a,  0xce3, 0x950, 0xb01, 0x105, 0xc57, 0x516, 0x63,  0xac,  0x43b, 0x9bb, 0x74b, 0x6af,
    0xca2, 0x4f4, 0x9f4, 0x39d, 0xc20, 0x1bf, 0xcba, 0x89f, 0xc,   0x491, 0x196, 0x272, 0x666, 0x359, 0xa8,  0x9a9,
    0x455, 0x36c, 0x474, 0x9c5, 0x4da, 0x47b, 0xa4,  0x48d, 0x594, 0x31d, 0x9b2, 0x30f, 0x887, 0x994, 0xbcd, 0x816,
    0x23,  0x180, 0xcd0, 0x673, 0x88a, 0x6ba, 0x3b4, 0xa9e, 0x73,  0x9fa, 0x23d, 0x48f, 0xc50, 0x4af, 0x54c, 0xad4,
    0xacb, 0xc7d, 0xa4b, 0xa27, 0x1f3, 0xbfe, 0x185, 0x6a1, 0x446, 0x7e7, 0x9b1, 0x3ed, 0x138, 0xc24, 0x374, 0x8ae,
    0x258, 0x26a, 0x472, 0x68e, 0x3,   0x4df, 0x369, 0x5a5, 0x36b, 0x428, 0xc02, 0x863, 0xb36, 0x498, 0x23d, 0x639,
    0x5ca, 0xd7,  0x46e, 0x5d8, 0xc72, 0x82a, 0x390, 0x947, 0xb4f, 0x9e9, 0xb19, 0x266, 0x9cc, 0xcdf, 0x704, 0x811,
  };
  const uint8_t reference_result[12][384] = {
    {0xca, 0x84, 0xfb, 0x10, 0x23, 0x4d, 0x42, 0x63, 0x8f, 0x3f, 0xe1, 0xb0, 0x15, 0x7, 0x1d, 0xa, 0x50, 0xac, 0x27,
     0xf2, 0x8a, 0xb2, 0xb7, 0xb5, 0x78, 0x68, 0x80, 0xcf, 0xec, 0xab, 0xed, 0xc0, 0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0, 0x0,  0x0, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x88, 0xd0, 0xd1, 0xb4, 0x5a, 0xea, 0xc,  0x75, 0xdb, 0xc5, 0xe1, 0x2c, 0x3f, 0x25, 0x56, 0x18, 0xaa, 0xd0, 0xbf,
     0x4e, 0xf2, 0x7b, 0x1d, 0xba, 0xd3, 0x1e, 0x56, 0x0,  0x63, 0x1f, 0xf9, 0x41, 0x30, 0x31, 0xb5, 0xc8, 0x1b, 0x3c,
     0x8,  0xad, 0x78, 0xc4, 0x44, 0xc6, 0xd5, 0x45, 0x76, 0x8f, 0x80, 0xdb, 0x5c, 0xe4, 0xf3, 0x81, 0x79, 0xd0, 0x95,
     0x94, 0xc5, 0x97, 0x92, 0xdc, 0x7f, 0xa3, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0xe0, 0x89, 0xc8, 0xf9, 0x7e, 0x9d, 0x63, 0xc4, 0xd2, 0x30, 0xb0, 0x38, 0x66, 0x2c, 0xe5, 0xcb, 0xc,  0x37, 0xaf,
     0x13, 0x2d, 0x9d, 0x84, 0x2a, 0x6c, 0x19, 0xa4, 0xb5, 0x49, 0x43, 0x84, 0xed, 0x56, 0xb1, 0xc0, 0x9e, 0x86, 0xcc,
     0x7,  0x9c, 0x12, 0x0,  0xe,  0x67, 0x7,  0x9a, 0xad, 0x43, 0xf8, 0xaf, 0xfb, 0x89, 0x9,  0xa2, 0xae, 0x0,  0x3f,
     0x18, 0x30, 0x93, 0x98, 0xf5, 0xa0, 0x58, 0x44, 0xc1, 0xd3, 0xbc, 0x61, 0x93, 0x55, 0xbf, 0x8,  0x58, 0xea, 0x70,
     0x86, 0xed, 0x87, 0x9d, 0x87, 0xab, 0x95, 0xab, 0xc9, 0x88, 0x69, 0xda, 0xfb, 0x85, 0xcc, 0x88, 0xca, 0xf7, 0x63,
     0xb0, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x81, 0x9e, 0x11, 0xb5, 0xf2, 0xd5, 0x4f, 0x8e, 0x77, 0x52, 0x98, 0xb8, 0xc0, 0x1,  0x26, 0x3d, 0x7c, 0xd2, 0x53,
     0xd1, 0x26, 0xc7, 0xc0, 0x2a, 0xad, 0x1d, 0x42, 0x17, 0x59, 0x44, 0x90, 0x24, 0xa9, 0x89, 0xf1, 0xb3, 0xcb, 0x8b,
     0xc7, 0x40, 0x18, 0xdc, 0x9c, 0x4b, 0xd3, 0x4,  0x99, 0x7d, 0xfb, 0xd4, 0xe8, 0x13, 0x58, 0x35, 0x11, 0x10, 0x2c,
     0x57, 0xdb, 0x2,  0x73, 0xcd, 0xe4, 0x3f, 0xf0, 0xee, 0xf4, 0xfb, 0x33, 0x8b, 0x80, 0xb0, 0x9c, 0x5,  0xb0, 0x1e,
     0x6f, 0x10, 0xc5, 0x89, 0x60, 0x4c, 0x2f, 0xb0, 0x60, 0x32, 0x48, 0xc1, 0x45, 0xc5, 0x66, 0x61, 0x47, 0x4c, 0xca,
     0xaf, 0x20, 0x80, 0x8b, 0xd5, 0xc1, 0x63, 0x6f, 0xd7, 0xfd, 0xcd, 0xf2, 0x82, 0xa5, 0x5c, 0xf1, 0xb4, 0x33, 0x85,
     0x60, 0x74, 0x54, 0xaf, 0x6e, 0x83, 0x17, 0x75, 0xaf, 0xb4, 0xce, 0x3e, 0xc,  0xa9, 0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x2,  0x76, 0x29, 0x44, 0xb2, 0xc4, 0xab, 0xdd, 0xf,  0x87, 0xce, 0x15, 0xf5, 0x24, 0xbc, 0x21, 0x8b, 0xc0, 0x8a,
     0x2e, 0xd7, 0x11, 0x7d, 0xd2, 0xd0, 0x6c, 0x34, 0xc,  0xf2, 0x1c, 0xba, 0xe6, 0x51, 0x4e, 0x13, 0x72, 0xa1, 0xf3,
     0x65, 0x1a, 0x92, 0xca, 0x37, 0xbe, 0xa9, 0xf6, 0xda, 0xf7, 0xee, 0x3f, 0x30, 0x64, 0x9d, 0x67, 0x45, 0x46, 0x1f,
     0x10, 0xe7, 0x7e, 0xd7, 0xa3, 0x1c, 0xb7, 0x9,  0x6f, 0x25, 0x33, 0x2,  0x8,  0x98, 0xbc, 0x65, 0x75, 0x9,  0xc7,
     0x69, 0x7c, 0xb8, 0x3f, 0xa0, 0xef, 0x9e, 0xfc, 0xed, 0xa6, 0xd8, 0x17, 0x20, 0xb0, 0x77, 0xa6, 0x0,  0xee, 0x1e,
     0xbe, 0x5,  0xa1, 0xb0, 0x84, 0x9f, 0xe1, 0xe4, 0xc9, 0xaf, 0x60, 0x11, 0x3,  0x91, 0xc0, 0xb,  0x2d, 0xcc, 0x96,
     0x58, 0xe,  0x61, 0x54, 0x71, 0xa7, 0x80, 0x0,  0x58, 0x63, 0xd2, 0x21, 0x9b, 0xe5, 0x59, 0xdb, 0xfb, 0xe7, 0x5c,
     0x3c, 0x81, 0x6b, 0x62, 0x35, 0x3c, 0xaa, 0xc6, 0x2c, 0x8,  0x18, 0x72, 0x48, 0xf9, 0xca, 0x97, 0x79, 0x4e, 0x2c,
     0xf7, 0x69, 0xba, 0x1c, 0x6f, 0x83, 0x41, 0xa4, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0xc4, 0x97, 0x8f, 0x44, 0x21, 0xb1, 0xc9, 0x5e, 0xd5, 0xbb, 0x83, 0x87, 0x1b, 0x97, 0x4c, 0x1f, 0xf9, 0xb5, 0x41,
     0x4c, 0x8,  0x57, 0x32, 0x2f, 0x2e, 0x97, 0xd0, 0x8d, 0x64, 0xd4, 0xd7, 0xb1, 0xbd, 0x81, 0x7c, 0x1a, 0x75, 0x2a,
     0x1b, 0xca, 0xb3, 0x15, 0xa4, 0xf5, 0x3c, 0x7f, 0x29, 0x19, 0x24, 0x5a, 0x7a, 0x45, 0xcf, 0xac, 0xab, 0xdb, 0x7e,
     0xde, 0xfb, 0x3f, 0xe0, 0x20, 0xcf, 0xb2, 0xa9, 0x42, 0xb,  0xfd, 0x0,  0x63, 0x59, 0x7b, 0x2e, 0x1f, 0xcd, 0xa1,
     0xcd, 0xc,  0x9e, 0x25, 0x31, 0x86, 0x0,  0x8,  0x30, 0xe2, 0x59, 0xed, 0xac, 0x4,  0xce, 0x46, 0xc3, 0x4f, 0xde,
     0x3b, 0x80, 0x6e, 0xeb, 0x11, 0xef, 0xea, 0x8b, 0xc2, 0x7a, 0xc2, 0xf7, 0xaf, 0xaf, 0x39, 0x9,  0xbf, 0x6b, 0x17,
     0x7d, 0x26, 0xc,  0x15, 0x4c, 0x86, 0x3e, 0x16, 0x4b, 0x7c, 0xf2, 0xab, 0x80, 0x85, 0x30, 0x1f, 0x34, 0xc0, 0x55,
     0x64, 0xc1, 0x98, 0x35, 0x58, 0xdb, 0x3,  0x3f, 0xea, 0xab, 0xa3, 0xc1, 0xf1, 0x83, 0x6a, 0x28, 0xd1, 0x42, 0xbc,
     0x58, 0xfd, 0xa5, 0xd5, 0x75, 0x3f, 0xcb, 0xca, 0x7e, 0x84, 0xd5, 0x9,  0x4f, 0x6,  0x1f, 0xad, 0xc,  0x63, 0x81,
     0x0,  0x16, 0x71, 0x11, 0xb5, 0xa7, 0xf7, 0xb5, 0x7c, 0x1c, 0x61, 0x75, 0x3d, 0x2a, 0xb9, 0x78, 0x7c, 0x33, 0xf0,
     0x3f, 0xa2, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x7,  0x9f, 0xfc, 0x88, 0x50, 0x94, 0xb2, 0x91, 0x7b, 0x6a, 0x5d, 0xe7, 0xc0, 0x85, 0xb7, 0xdc, 0xe4, 0xe4, 0x4b,
     0xfa, 0xb4, 0x82, 0xf1, 0x81, 0xf0, 0x9a, 0x9c, 0x2d, 0x5d, 0x5c, 0xe4, 0xbc, 0x29, 0x2d, 0xd4, 0x2e, 0x87, 0xed,
     0x2b, 0x20, 0x37, 0x1b, 0x69, 0x29, 0x79, 0x31, 0xf1, 0xd8, 0x12, 0x48, 0x96, 0xa7, 0xd3, 0x4f, 0x8e, 0x1a, 0x47,
     0x68, 0xb2, 0xb7, 0xd8, 0x63, 0xaa, 0x56, 0xae, 0xb6, 0xc7, 0xeb, 0xf6, 0x3b, 0x41, 0x3,  0xf9, 0x4c, 0x5e, 0x52,
     0x41, 0x96, 0xb3, 0x7,  0x50, 0x5c, 0xaa, 0x77, 0x5c, 0xbc, 0xc8, 0x2c, 0x6c, 0x63, 0xc,  0xbd, 0x55, 0x9,  0xc3,
     0x20, 0x0,  0x8,  0xdf, 0x7,  0x8f, 0xa5, 0x35, 0x53, 0x6,  0x9b, 0xdb, 0xf9, 0xdb, 0x89, 0xeb, 0x37, 0x80, 0x3a,
     0x7b, 0x2e, 0xc2, 0x73, 0xe9, 0x16, 0xa,  0x96, 0x37, 0xf0, 0xf9, 0xaf, 0x5d, 0x66, 0x69, 0xf0, 0xe7, 0xb2, 0x15,
     0x79, 0x19, 0xe1, 0xa0, 0x2,  0x23, 0x85, 0xfc, 0x98, 0x98, 0x74, 0x8f, 0xf4, 0xab, 0x80, 0x16, 0x4,  0xf3, 0xb,
     0x19, 0xbe, 0x2b, 0x11, 0xb,  0xc,  0x63, 0x19, 0x5a, 0xb7, 0xcf, 0xd7, 0x43, 0xf5, 0xd2, 0xa1, 0x81, 0x87, 0xff,
     0x47, 0x15, 0x92, 0xd2, 0x4,  0xb1, 0xa5, 0x95, 0x77, 0xd1, 0xd6, 0xea, 0x7d, 0x99, 0x3c, 0xb1, 0x3f, 0x82, 0x2a,
     0xe7, 0xf7, 0xc4, 0xc0, 0x8b, 0xaa, 0x17, 0xc,  0x2b, 0x8,  0x80, 0x89, 0x70, 0xa2, 0x94, 0x7d, 0xea, 0x6e, 0x59,
     0x7a, 0x39, 0x4,  0x4b, 0xb7, 0x87, 0x8e, 0xb6, 0x6f, 0x71, 0x1b, 0x3,  0xfe, 0x17, 0x9f, 0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0xf,  0x7d, 0xe4, 0x8d, 0xf,  0x14, 0x49, 0xb2, 0x23, 0xee, 0x52, 0xd6, 0xeb, 0x38, 0xdf, 0x83, 0x6d, 0x71, 0x26,
     0x4d, 0x7c, 0x91, 0x7d, 0xb4, 0x4,  0xc6, 0xf,  0x7,  0x5d, 0x26, 0xcd, 0x2c, 0xba, 0x71, 0x22, 0xce, 0x36, 0x4a,
     0x17, 0xd3, 0x5c, 0x1c, 0x6c, 0xbd, 0x4,  0xc7, 0x9a, 0x1a, 0xd3, 0xa4, 0xc9, 0x16, 0x27, 0x3c, 0x6c, 0x12, 0x90,
     0x57, 0x3d, 0x3a, 0xfb, 0x92, 0x46, 0x19, 0x8e, 0xa0, 0x93, 0x7a, 0x15, 0xf5, 0x2f, 0xaa, 0xac, 0xb8, 0xb4, 0x7b,
     0x77, 0xbb, 0xfa, 0x3b, 0x81, 0xb,  0xc8, 0xce, 0xc7, 0x97, 0xa8, 0x40, 0x2d, 0xce, 0x3c, 0x0,  0x8b, 0x96, 0xd5,
     0x76, 0xb7, 0xf0, 0x44, 0xcb, 0x84, 0xd9, 0x30, 0xc,  0x7a, 0x56, 0x4a, 0x30, 0x17, 0x9,  0x0,  0x9,  0xbe, 0x1f,
     0x77, 0x58, 0xb4, 0xcc, 0x28, 0x6,  0x37, 0x6e, 0xce, 0xbf, 0x3b, 0xe2, 0xf4, 0x36, 0x1,  0xea, 0xd8, 0xe6, 0x44,
     0xef, 0xb7, 0xe8, 0x2c, 0x29, 0xb0, 0x79, 0x7,  0x7d, 0xfd, 0xad, 0xba, 0x97, 0x4a, 0x7,  0xfe, 0xb7, 0xd9, 0x14,
     0xf3, 0x64, 0x8,  0xd,  0x53, 0xc0, 0x90, 0x84, 0xf9, 0x62, 0xc4, 0x47, 0xef, 0x22, 0xfb, 0xaa, 0x1,  0x5a, 0x1f,
     0x30, 0x7e, 0x42, 0xd,  0xbe, 0x55, 0x43, 0x58, 0xc0, 0x60, 0x58, 0xd,  0x5a, 0x6e, 0x3d, 0xbf, 0x3c, 0xa8, 0xbd,
     0xe8, 0x9f, 0x3,  0x1e, 0xfc, 0x7f, 0xa8, 0x84, 0x49, 0xd1, 0x9,  0xc4, 0x2c, 0x5a, 0xf2, 0x5c, 0x68, 0xd5, 0xd4,
     0xf6, 0xcb, 0xc8, 0x26, 0xec, 0x1e, 0x82, 0x54, 0x9c, 0xbf, 0x4d, 0x18, 0xef, 0x44, 0xab, 0x2e, 0x30, 0x58, 0x81,
     0x0,  0x60, 0x43, 0x6f, 0x43, 0x52, 0xec, 0xa5, 0xdd, 0x5a, 0x2c, 0x7b, 0x72, 0x11, 0x57, 0x73, 0xf5, 0xa1, 0x46,
     0xb7, 0xdf, 0xc3, 0xda, 0x2f, 0xc1, 0xfd, 0x8a, 0x9f, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x1d, 0xf2, 0x21, 0xd7, 0xf8, 0x1,  0xc5, 0xa4, 0xb1, 0x46, 0xb6, 0x93, 0x5a, 0x6d, 0x3d, 0xce, 0xef, 0x83, 0xdb,
     0xc6, 0x31, 0xd1, 0x84, 0x6f, 0xa4, 0x3e, 0xb4, 0x9,  0x18, 0x77, 0x70, 0xa0, 0x6b, 0x89, 0x66, 0x2c, 0x74, 0xc5,
     0x15, 0xe9, 0xcc, 0x86, 0x92, 0x8b, 0xd3, 0xb9, 0x70, 0x60, 0xd3, 0x7b, 0xc0, 0x71, 0x4d, 0x1a, 0xa6, 0x91, 0x4a,
     0x66, 0xd1, 0xe4, 0xce, 0x35, 0x12, 0x20, 0x5d, 0xe5, 0xa9, 0x63, 0xbf, 0x64, 0x23, 0x19, 0x1d, 0x7f, 0x96, 0xa4,
     0xb7, 0x42, 0xbd, 0x17, 0xaa, 0x58, 0xe3, 0xa2, 0xb5, 0xe7, 0xae, 0x2e, 0x7d, 0x3b, 0x2,  0x2d, 0x3c, 0xde, 0xfc,
     0xd8, 0x65, 0x54, 0x40, 0x5a, 0x38, 0xe3, 0x1,  0x60, 0x71, 0x65, 0xea, 0x76, 0x6e, 0xc1, 0x1f, 0xba, 0x8c, 0x50,
     0x36, 0x98, 0xb,  0xf3, 0x5a, 0x4d, 0xfa, 0xf2, 0x42, 0x2,  0x0,  0x9,  0x7d, 0x7d, 0xbc, 0x83, 0x85, 0x16, 0x33,
     0x14, 0x6,  0x6e, 0xb6, 0x71, 0xee, 0x5b, 0xa7, 0x38, 0xfa, 0x36, 0x1,  0xa6, 0xc7, 0x6e, 0x9e, 0xe8, 0xbb, 0x5b,
     0xe8, 0x59, 0xa4, 0x84, 0x95, 0xe7, 0x40, 0x5f, 0xfe, 0xad, 0x74, 0x5d, 0x56, 0x72, 0xb0, 0xff, 0x6d, 0x6c, 0x14,
     0xe6, 0x91, 0x3d, 0xd0, 0x70, 0xea, 0xef, 0xc7, 0x83, 0xf1, 0x87, 0x21, 0x76, 0xd4, 0xbd, 0x48, 0xfd, 0xa9, 0x2,
     0x68, 0xf9, 0x0,  0xc3, 0x8f, 0x90, 0x6,  0xbe, 0xab, 0xe,  0xbd, 0xa,  0xfc, 0xb,  0x56, 0x86, 0x59, 0xdc, 0xf6,
     0xf8, 0xc5, 0x3,  0x35, 0x6f, 0x74, 0x9f, 0x5,  0x76, 0xe0, 0xf7, 0x7,  0x35, 0xa1, 0x24, 0xd1, 0x12, 0x12, 0x63,
     0x99, 0x55, 0x1e, 0x57, 0x34, 0xd5, 0xa9, 0xd9, 0x57, 0x86, 0xdc, 0x4,  0x3b, 0x8f, 0x82, 0xa8, 0x6e, 0xfa, 0xdd,
     0x4,  0xc3, 0x3b, 0x22, 0xab, 0x5c, 0xbe, 0xbc, 0x12, 0x8,  0x0,  0x98, 0x21, 0x6f, 0x87, 0x48, 0x65, 0x57, 0x9a,
     0xbb, 0x16, 0x96, 0x7a, 0xe4, 0x42, 0xb8, 0x32, 0xa7, 0x3e, 0x28, 0xa3, 0xb6, 0xbd, 0xd,  0xd7, 0xf6, 0x22, 0x78,
     0x3f, 0x45, 0x9f, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x3b, 0xcc, 0x17, 0x79, 0x8d, 0x3e, 0x40, 0x51, 0xd2, 0xb1, 0x8b, 0xd8, 0x9e, 0x94, 0xd5, 0xab, 0x87, 0xe3, 0xb7,
     0x83, 0xb5, 0x15, 0x77, 0x49, 0x4d, 0xf0, 0x19, 0x49, 0x1f, 0xb4, 0x12, 0x5c, 0xbc, 0x3,  0x7,  0x75, 0x59, 0x52,
     0xf3, 0x2b, 0xe7, 0x12, 0x97, 0x88, 0xce, 0xd8, 0xa0, 0xc4, 0x45, 0xd3, 0x72, 0xc1, 0xf1, 0x1a, 0xbd, 0xe,  0x70,
     0x9c, 0xe6, 0x19, 0x4c, 0x47, 0x3a, 0x72, 0x16, 0x9a, 0xbc, 0xf3, 0x1a, 0x12, 0x41, 0x72, 0x35, 0x4f, 0x3a, 0xec,
     0x27, 0x99, 0x11, 0x19, 0x39, 0xfa, 0xb9, 0x24, 0x7a, 0x56, 0x50, 0xdf, 0xb,  0xaa, 0xb1, 0x8a, 0xb,  0xed, 0x7a,
     0xdc, 0xad, 0x8b, 0xfe, 0x3a, 0x4,  0xb6, 0xe0, 0xf1, 0xcd, 0x1e, 0x6f, 0x29, 0x6a, 0x40, 0xb3, 0xe4, 0xfc, 0x4e,
     0x0,  0x2c, 0x5e, 0x39, 0xb5, 0x76, 0xdc, 0x6,  0xff, 0x50, 0xcb, 0x11, 0x92, 0xd,  0xcc, 0xb,  0xe6, 0x65, 0x75,
     0x92, 0x2f, 0x5e, 0x8c, 0x0,  0xc0, 0x8,  0xfa, 0xee, 0xd1, 0x1d, 0x58, 0xcf, 0xc2, 0xc,  0xca, 0x5,  0xdb, 0xdc,
     0x96, 0xb3, 0xbe, 0xea, 0x28, 0xfe, 0x7c, 0x36, 0x2,  0x9c, 0x2e, 0xb6, 0xe6, 0x11, 0xf5, 0xce, 0x2d, 0xe8, 0xb1,
     0x90, 0x12, 0x2c, 0x79, 0x1c, 0xd0, 0x27, 0xbf, 0xad, 0xe8, 0x76, 0x99, 0x12, 0x7,  0xf7, 0x77, 0x3b, 0x36, 0x14,
     0xcc, 0x43, 0xe6, 0x41, 0xd,  0x4d, 0xf9, 0xeb, 0xa3, 0x83, 0xe3, 0x1b, 0x6,  0x71, 0x47, 0xbb, 0x27, 0xa2, 0xfe,
     0xa9, 0x4,  0xa0, 0xd5, 0x47, 0x30, 0xf8, 0x21, 0x44, 0x43, 0xbe, 0x55, 0x35, 0xf4, 0x55, 0xc0, 0x7e, 0x85, 0x25,
     0x83, 0x59, 0xb7, 0xd5, 0xb3, 0x6f, 0x3c, 0x9f, 0xca, 0x1b, 0x7a, 0x9f, 0xb,  0xd8, 0x11, 0x3f, 0x7f, 0xa0, 0x4a,
     0x48, 0x12, 0xd1, 0x23, 0x48, 0xc,  0xcb, 0x59, 0xca, 0xc7, 0x15, 0x5a, 0xd5, 0x52, 0x5f, 0xbf, 0xf2, 0xc7, 0x99,
     0xc0, 0x8e, 0x87, 0x82, 0x51, 0xb9, 0xb9, 0x6f, 0x4d, 0x60, 0xf0, 0xe,  0xd1, 0xaa, 0xb9, 0xf8, 0xe2, 0x15, 0x81,
     0x1,  0x0,  0xd6, 0x10, 0x6f, 0xd,  0x1d, 0x25, 0x3b, 0xa5, 0x73, 0xab, 0x5,  0x8b, 0x7a, 0xc8, 0x9,  0xd1, 0x15,
     0x73, 0xd4, 0xf,  0x9a, 0xd1, 0xb6, 0x7b, 0x33, 0xac, 0x76, 0x2f, 0x3,  0xdb, 0x8f, 0xe2, 0x9e, 0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0x76, 0x30, 0x5f, 0xc8, 0xd5, 0xb8, 0x87, 0x4f, 0x28, 0xa9, 0xb1, 0x16, 0x61, 0x7b, 0xa4, 0x58, 0x6d, 0xf5, 0xe1,
     0xec, 0x7b, 0x83, 0x6a, 0x53, 0x9c, 0x4b, 0xd4, 0xf4, 0x3d, 0x46, 0x9e, 0xf,  0xb4, 0x23, 0x78, 0xb1, 0x1d, 0x72,
     0xa0, 0xae, 0x96, 0xa8, 0xf9, 0x2b, 0xce, 0x3d, 0xdc, 0x44, 0xe8, 0xfc, 0x9a, 0x27, 0xdd, 0x62, 0xd3, 0xe3, 0xfa,
     0xc6, 0xd7, 0xd0, 0xcb, 0x81, 0x1c, 0x4b, 0xf3, 0x19, 0x97, 0xe,  0xe9, 0x91, 0x63, 0x41, 0x93, 0xee, 0x78, 0x2d,
     0x12, 0x81, 0xc4, 0x55, 0x79, 0xa4, 0x73, 0xfd, 0x49, 0xca, 0x28, 0x19, 0x73, 0xec, 0xa7, 0x25, 0x9f, 0xc7, 0x8a,
     0xd4, 0xe7, 0x25, 0xaa, 0x61, 0x25, 0x2e, 0x68, 0xad, 0x87, 0x3b, 0xeb, 0x42, 0xff, 0x3a, 0x9,  0xd4, 0x2,  0x8f,
     0xdb, 0xcc, 0x63, 0x5b, 0x12, 0x55, 0x40, 0x67, 0x91, 0xf3, 0x77, 0x2,  0x80, 0x45, 0x57, 0x96, 0x7a, 0x76, 0xb8,
     0x15, 0xbc, 0x87, 0xb6, 0x2c, 0x42, 0x64, 0x3,  0xc6, 0xb,  0xcd, 0x93, 0x95, 0x93, 0xfa, 0xc2, 0x8b, 0x23, 0x4,
     0xc0, 0x8,  0xf4, 0xbd, 0x87, 0xee, 0x80, 0xe5, 0xd9, 0x30, 0x3,  0xe5, 0x5,  0xb7, 0x69, 0x9b, 0x9c, 0xe9, 0x4b,
     0x1d, 0x8a, 0x7f, 0x5e, 0x36, 0x5,  0x70, 0xfa, 0xb0, 0x67, 0x3e, 0xa2, 0xbd, 0xe7, 0x36, 0xe8, 0x63, 0x39, 0x8a,
     0x60, 0x91, 0x87, 0x3,  0xf4, 0x91, 0x9f, 0xad, 0xd1, 0xcd, 0xa5, 0x94, 0x6e, 0xe0, 0xfe, 0xdd, 0x16, 0x3b, 0x14,
     0x97, 0xf,  0x59, 0xf,  0xd4, 0xa0, 0x29, 0xfe, 0xf6, 0xb1, 0x83, 0xc6, 0x67, 0x18, 0x88, 0x73, 0x64, 0xf7, 0x89,
     0x50, 0xdf, 0xa9, 0x7,  0x78, 0x96, 0x3e, 0x2,  0x3,  0xbf, 0x7,  0x9d, 0x21, 0xbe, 0xaa, 0xda, 0x50, 0xaf, 0x6,
     0xcc, 0x2f, 0x61, 0x95, 0xa1, 0x59, 0x6f, 0x53, 0xcf, 0x7d, 0xc5, 0xf3, 0x53, 0xf2, 0xe,  0x3d, 0x9f, 0x16, 0x60,
     0x87, 0xf8, 0xf1, 0x17, 0xd4, 0x11, 0x1e, 0x9,  0xd1, 0x47, 0x18, 0x71, 0x58, 0x9c, 0x35, 0x79, 0x71, 0x9,  0x2d,
     0xd5, 0xa4, 0x7e, 0x7d, 0x95, 0x7f, 0x3c, 0x93, 0xb0, 0xbf, 0x83, 0x82, 0xa1, 0xea, 0xa6, 0x7d, 0xd5, 0x4,  0xc,
     0xbc, 0x83, 0xe8, 0xaa, 0x71, 0xe1, 0xb,  0xaf, 0x10, 0x28, 0x80, 0x7f, 0x65, 0x28, 0x6f, 0x1a, 0x7a, 0xd4, 0xd8,
     0x53, 0x6a, 0xee, 0x69, 0x85, 0x85, 0x7a, 0x90, 0x23, 0x84, 0xae, 0x30, 0x87, 0x7a, 0x83, 0xc6, 0xa8, 0xb6, 0xf5,
     0xce, 0x30, 0xb5, 0xf5, 0x72, 0xe0, 0xf5, 0x47, 0xd1, 0x9e, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
     0x0,  0x0,  0x0,  0x0},
    {0xbf, 0x60, 0x65, 0x97, 0xcb, 0x72, 0xc8, 0x30, 0x10, 0xb9, 0x63, 0x90, 0xc4, 0x11, 0xc1, 0x2c, 0x94, 0xad, 0xed,
     0xdb, 0x2d, 0x59, 0xdb, 0x6a, 0x8d, 0x15, 0x5c, 0xeb, 0xc1, 0x3e, 0x4b, 0x46, 0x76, 0x58, 0x56, 0x92, 0x39, 0xd0,
     0xa0, 0xbf, 0xc0, 0x5,  0xbc, 0x94, 0x1e, 0x6d, 0xaa, 0x23, 0x70, 0xc9, 0x5b, 0xbf, 0xd1, 0xa7, 0xbd, 0x12, 0x3c,
     0x2a, 0xc1, 0xab, 0xb2, 0xb4, 0x16, 0x7a, 0xa5, 0x99, 0x2e, 0xd0, 0xa1, 0xd6, 0x7,  0x15, 0xb7, 0x7a, 0x85, 0x34,
     0xa,  0x12, 0xf5, 0x81, 0x30, 0x79, 0xb5, 0xe,  0x53, 0xc7, 0x46, 0x15, 0x63, 0x2f, 0xbf, 0xfc, 0x76, 0x92, 0x63,
     0x14, 0x3b, 0xb7, 0x81, 0x74, 0x17, 0x63, 0x17, 0x31, 0xc7, 0x65, 0x42, 0x8a, 0xbf, 0xf8, 0x95, 0x24, 0xd9, 0x63,
     0xc,  0xc6, 0x97, 0xb3, 0xdc, 0x2f, 0x8f, 0x26, 0x9,  0x23, 0x2a, 0xa7, 0x22, 0x9a, 0x7a, 0x8f, 0x38, 0x34, 0x47,
     0xa2, 0xa7, 0xa,  0x23, 0x0,  0xf,  0xb7, 0x79, 0xcd, 0x4a, 0x60, 0x4c, 0x49, 0xc3, 0x71, 0x43, 0xa5, 0xb7, 0x66,
     0xb0, 0x70, 0x82, 0x9,  0x2d, 0x26, 0x46, 0xbf, 0xb3, 0x26, 0x31, 0x31, 0x7,  0x1,  0x20, 0x7,  0xad, 0x19, 0x19,
     0xf,  0x86, 0x47, 0x22, 0xe9, 0xa5, 0x8,  0xc2, 0x4,  0xc9, 0x22, 0x59, 0x7a, 0xea, 0x9a, 0xf9, 0x12, 0xb8, 0x63,
     0x2c, 0x2c, 0x8,  0xf0, 0xbd, 0xfe, 0x4a, 0xbb, 0x79, 0x83, 0xc2, 0x4d, 0xa9, 0xbc, 0x41, 0x42, 0x21, 0xf4, 0x58,
     0x62, 0x5b, 0x90, 0x65, 0xd4, 0xc,  0x8d, 0x74, 0xe9, 0x7a, 0xc6, 0xa3, 0x5,  0xe3, 0xc,  0x95, 0x1,  0x5b, 0x10,
     0x57, 0x6c, 0x51, 0x63, 0xc0, 0xa,  0x3b, 0xb4, 0x9b, 0x4b, 0xf7, 0x6a, 0xa2, 0x4c, 0x4f, 0xf4, 0xd9, 0x39, 0x20,
     0xfc, 0x1b, 0xba, 0xfc, 0x89, 0xc,  0x10, 0x49, 0x96, 0x21, 0x27, 0x66, 0x96, 0x35, 0xa8, 0x90, 0x9a, 0x55, 0xc4,
     0x36, 0x74, 0x54, 0x9c, 0xda, 0xb4, 0x47, 0xa4, 0xd0, 0x48, 0x94, 0xd5, 0x31, 0xb2, 0xf9, 0x30, 0x87, 0x48, 0x99,
     0xcd, 0x6b, 0x81, 0x23, 0x0,  0x18, 0xd0, 0x3c, 0x67, 0x8a, 0xa8, 0x6b, 0xb4, 0xe3, 0xa9, 0x73, 0xa0, 0x9f, 0x3d,
     0xf2, 0x48, 0x50, 0xfc, 0x4a, 0x4c, 0x45, 0xad, 0xcb, 0xda, 0xc7, 0x4b, 0x7a, 0xa2, 0xf3, 0xe1, 0xbf, 0x85, 0x11,
     0x6a, 0x46, 0x74, 0x7e, 0xb1, 0xd9, 0x3e, 0x38, 0x41, 0xc2, 0x74, 0xe3, 0x8a, 0x58, 0xa2, 0x26, 0x72, 0xe4, 0x68,
     0x3,  0xf0, 0x4d, 0x69, 0x53, 0x5a, 0x6b, 0x83, 0x42, 0x2,  0x3c, 0x86, 0x36, 0x8b, 0x49, 0x3d, 0x92, 0x63, 0xca,
     0x75, 0xd,  0x6e, 0x84, 0x5d, 0x72, 0xac, 0x82, 0x90, 0x73, 0x94, 0x4f, 0x9b, 0x9e, 0x19, 0x6b, 0x26, 0xcc, 0xf9,
     0xcd, 0x4,  0x17, 0x81},
  };
  const uint16_t reference_decompressed[12][256] = {
    {0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x681, 0x681, 0x0,   0x0,   0x681, 0x0,   0x0,   0x0,   0x0,   0x681,
     0x681, 0x681, 0x0,   0x681, 0x681, 0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x0,   0x681, 0x0,   0x0,   0x0,
     0x681, 0x681, 0x0,   0x0,   0x0,   0x681, 0x0,   0x0,   0x681, 0x0,   0x681, 0x681, 0x0,   0x0,   0x681, 0x0,
     0x0,   0x681, 0x0,   0x0,   0x0,   0x0,   0x681, 0x0,   0x681, 0x681, 0x0,   0x0,   0x0,   0x681, 0x681, 0x0,
     0x681, 0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x681, 0x681, 0x681, 0x681, 0x681, 0x681, 0x681, 0x0,   0x0,
     0x681, 0x0,   0x0,   0x0,   0x0,   0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x0,   0x681, 0x681, 0x0,   0x681,
     0x681, 0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x0,   0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x0,   0x0,
     0x681, 0x0,   0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x0,   0x0,
     0x0,   0x0,   0x0,   0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x0,   0x681, 0x681, 0x0,   0x681, 0x0,   0x681,
     0x681, 0x681, 0x681, 0x0,   0x0,   0x681, 0x0,   0x0,   0x0,   0x681, 0x0,   0x0,   0x681, 0x681, 0x681, 0x681,
     0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x681, 0x681, 0x0,   0x681,
     0x681, 0x681, 0x681, 0x0,   0x681, 0x681, 0x0,   0x681, 0x681, 0x0,   0x681, 0x0,   0x681, 0x681, 0x0,   0x681,
     0x0,   0x0,   0x0,   0x681, 0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x0,   0x681, 0x0,   0x681, 0x681, 0x0,
     0x0,   0x0,   0x0,   0x0,   0x0,   0x0,   0x0,   0x681, 0x681, 0x681, 0x681, 0x681, 0x0,   0x0,   0x681, 0x681,
     0x0,   0x0,   0x681, 0x681, 0x0,   0x681, 0x681, 0x681, 0x681, 0x681, 0x0,   0x681, 0x0,   0x681, 0x0,   0x681,
     0x681, 0x0,   0x681, 0x681, 0x0,   0x681, 0x681, 0x681, 0x0,   0x0,   0x0,   0x0,   0x0,   0x0,   0x681, 0x681},
    {0x0,   0x681, 0x0,   0x681, 0x0,   0x0,   0x340, 0x9c1, 0x340, 0x0,   0x340, 0x9c1, 0x0,   0x340, 0x9c1, 0x681,
     0x681, 0x681, 0x340, 0x340, 0x681, 0x681, 0x681, 0x9c1, 0x0,   0x9c1, 0x0,   0x0,   0x340, 0x340, 0x9c1, 0x340,
     0x9c1, 0x681, 0x340, 0x9c1, 0x340, 0x340, 0x0,   0x9c1, 0x340, 0x0,   0x681, 0x9c1, 0x0,   0x9c1, 0x681, 0x0,
     0x9c1, 0x9c1, 0x9c1, 0x0,   0x340, 0x340, 0x681, 0x0,   0x681, 0x340, 0x340, 0x340, 0x0,   0x681, 0x340, 0x0,
     0x681, 0x681, 0x681, 0x681, 0x0,   0x0,   0x340, 0x9c1, 0x9c1, 0x9c1, 0x9c1, 0x681, 0x681, 0x9c1, 0x0,   0x340,
     0x681, 0x0,   0x9c1, 0x9c1, 0x9c1, 0x681, 0x9c1, 0x340, 0x340, 0x9c1, 0x340, 0x0,   0x681, 0x681, 0x9c1, 0x681,
     0x9c1, 0x0,   0x340, 0x9c1, 0x681, 0x9c1, 0x340, 0x0,   0x681, 0x340, 0x340, 0x340, 0x0,   0x0,   0x0,   0x0,
     0x9c1, 0x0,   0x681, 0x340, 0x9c1, 0x9c1, 0x340, 0x0,   0x340, 0x681, 0x9c1, 0x9c1, 0x340, 0x0,   0x0,   0x340,
     0x0,   0x0,   0x9c1, 0x0,   0x340, 0x0,   0x9c1, 0x0,   0x340, 0x340, 0x9c1, 0x681, 0x0,   0x681, 0x0,   0x9c1,
     0x9c1, 0x681, 0x340, 0x0,   0x0,   0x9c1, 0x9c1, 0x0,   0x0,   0x681, 0x0,   0x0,   0x340, 0x9c1, 0x681, 0x681,
     0x0,   0x681, 0x9c1, 0x340, 0x0,   0x340, 0x0,   0x9c1, 0x0,   0x340, 0x0,   0x340, 0x681, 0x340, 0x0,   0x9c1,
     0x340, 0x340, 0x340, 0x9c1, 0x340, 0x340, 0x0,   0x340, 0x681, 0x340, 0x9c1, 0x340, 0x9c1, 0x9c1, 0x0,   0x681,
     0x0,   0x0,   0x0,   0x681, 0x9c1, 0x681, 0x340, 0x9c1, 0x0,   0x9c1, 0x340, 0x340, 0x0,   0x340, 0x681, 0x9c1,
     0x9c1, 0x0,   0x9c1, 0x9c1, 0x340, 0x0,   0x0,   0x681, 0x340, 0x681, 0x9c1, 0x340, 0x0,   0x0,   0x340, 0x9c1,
     0x340, 0x340, 0x340, 0x681, 0x0,   0x340, 0x340, 0x681, 0x340, 0x340, 0x0,   0x9c1, 0x9c1, 0x340, 0x340, 0x681,
     0x681, 0x0,   0x340, 0x681, 0x0,   0x9c1, 0x340, 0x9c1, 0x9c1, 0x9c1, 0x9c1, 0x340, 0x9c1, 0x0,   0x681, 0x681},
    {0x0,   0x681, 0xb61, 0x681, 0x0,   0x1a0, 0x340, 0x9c1, 0x1a0, 0xb61, 0x4e0, 0xb61, 0xb61, 0x340, 0xb61, 0x681,
     0x4e0, 0x681, 0x1a0, 0x340, 0x681, 0x821, 0x681, 0x9c1, 0x0,   0x9c1, 0x0,   0x0,   0x4e0, 0x1a0, 0x9c1, 0x1a0,
     0x9c1, 0x681, 0x1a0, 0x9c1, 0x340, 0x340, 0x1a0, 0xb61, 0x4e0, 0x1a0, 0x4e0, 0x9c1, 0x0,   0x9c1, 0x821, 0x1a0,
     0xb61, 0x821, 0x9c1, 0x1a0, 0x1a0, 0x340, 0x4e0, 0x1a0, 0x821, 0x4e0, 0x340, 0x340, 0x0,   0x821, 0x340, 0x1a0,
     0x681, 0x821, 0x821, 0x681, 0x1a0, 0x0,   0x1a0, 0x821, 0x821, 0x9c1, 0x9c1, 0x681, 0x681, 0x9c1, 0x0,   0x340,
     0x681, 0x0,   0x9c1, 0x9c1, 0x9c1, 0x821, 0x821, 0x340, 0x1a0, 0x9c1, 0x340, 0x0,   0x681, 0x821, 0xb61, 0x681,
     0x9c1, 0x0,   0x340, 0x9c1, 0x681, 0xb61, 0x1a0, 0x0,   0x681, 0x4e0, 0x340, 0x1a0, 0x1a0, 0x0,   0x0,   0x0,
     0x9c1, 0x1a0, 0x681, 0x4e0, 0x9c1, 0x9c1, 0x1a0, 0x0,   0x340, 0x4e0, 0x9c1, 0x9c1, 0x340, 0xb61, 0x0,   0x340,
     0x0,   0xb61, 0xb61, 0xb61, 0x340, 0xb61, 0x9c1, 0xb61, 0x1a0, 0x1a0, 0x9c1, 0x681, 0x0,   0x681, 0x0,   0x821,
     0x9c1, 0x821, 0x340, 0x0,   0x0,   0x9c1, 0xb61, 0x1a0, 0x0,   0x4e0, 0x0,   0x0,   0x4e0, 0x9c1, 0x681, 0x681,
     0x0,   0x4e0, 0x9c1, 0x340, 0xb61, 0x1a0, 0x0,   0x821, 0x0,   0x4e0, 0x1a0, 0x340, 0x681, 0x340, 0x0,   0x9c1,
     0x4e0, 0x340, 0x4e0, 0x9c1, 0x4e0, 0x4e0, 0x0,   0x4e0, 0x4e0, 0x340, 0x9c1, 0x340, 0x821, 0x9c1, 0xb61, 0x821,
     0x0,   0x1a0, 0x0,   0x681, 0x821, 0x681, 0x340, 0xb61, 0x0,   0x9c1, 0x1a0, 0x4e0, 0x0,   0x4e0, 0x4e0, 0xb61,
     0xb61, 0x0,   0x9c1, 0x9c1, 0x1a0, 0xb61, 0x1a0, 0x681, 0x4e0, 0x821, 0x9c1, 0x340, 0x1a0, 0xb61, 0x340, 0x821,
     0x1a0, 0x1a0, 0x4e0, 0x681, 0x0,   0x4e0, 0x340, 0x4e0, 0x340, 0x4e0, 0xb61, 0x821, 0xb61, 0x4e0, 0x1a0, 0x681,
     0x681, 0x1a0, 0x4e0, 0x681, 0x0,   0x821, 0x340, 0x9c1, 0xb61, 0x9c1, 0xb61, 0x1a0, 0x9c1, 0x0,   0x681, 0x821},
    {0xd0,  0x681, 0xb61, 0x751, 0xd0,  0xd0,  0x410, 0x8f1, 0x1a0, 0xc31, 0x410, 0xa91, 0xc31, 0x340, 0xb61, 0x681,
     0x5b0, 0x5b0, 0x1a0, 0x410, 0x681, 0x751, 0x681, 0x8f1, 0x0,   0x9c1, 0xd0,  0x0,   0x4e0, 0x1a0, 0xa91, 0x270,
     0x9c1, 0x5b0, 0x1a0, 0xa91, 0x270, 0x410, 0xd0,  0xa91, 0x4e0, 0x1a0, 0x5b0, 0x9c1, 0x0,   0x9c1, 0x821, 0x1a0,
     0xa91, 0x821, 0xa91, 0xd0,  0x1a0, 0x340, 0x5b0, 0xd0,  0x751, 0x410, 0x340, 0x340, 0x0,   0x751, 0x340, 0x1a0,
     0x751, 0x821, 0x751, 0x681, 0xd0,  0xc31, 0x270, 0x8f1, 0x8f1, 0x9c1, 0x8f1, 0x681, 0x5b0, 0x9c1, 0x0,   0x340,
     0x681, 0xd0,  0x9c1, 0xa91, 0x9c1, 0x751, 0x8f1, 0x340, 0x270, 0xa91, 0x340, 0x0,   0x751, 0x751, 0xa91, 0x5b0,
     0x8f1, 0xc31, 0x340, 0xa91, 0x681, 0xb61, 0x270, 0xd0,  0x681, 0x410, 0x410, 0x270, 0xd0,  0xd0,  0x0,   0xd0,
     0x9c1, 0x1a0, 0x5b0, 0x410, 0x8f1, 0xa91, 0x1a0, 0x0,   0x270, 0x5b0, 0xa91, 0x9c1, 0x340, 0xb61, 0xc31, 0x270,
     0x0,   0xc31, 0xb61, 0xb61, 0x340, 0xc31, 0x8f1, 0xc31, 0x270, 0x270, 0x8f1, 0x681, 0x0,   0x681, 0x0,   0x8f1,
     0x9c1, 0x751, 0x410, 0x0,   0x0,   0x8f1, 0xb61, 0xd0,  0xc31, 0x4e0, 0x0,   0xd0,  0x410, 0x9c1, 0x751, 0x681,
     0x0,   0x4e0, 0x9c1, 0x340, 0xc31, 0x1a0, 0x0,   0x8f1, 0x0,   0x4e0, 0x1a0, 0x270, 0x681, 0x340, 0xd0,  0x9c1,
     0x410, 0x340, 0x410, 0x9c1, 0x4e0, 0x4e0, 0xd0,  0x4e0, 0x5b0, 0x340, 0x9c1, 0x340, 0x821, 0x9c1, 0xc31, 0x821,
     0x0,   0x1a0, 0x0,   0x681, 0x8f1, 0x681, 0x410, 0xa91, 0xd0,  0x9c1, 0x270, 0x4e0, 0xc31, 0x4e0, 0x5b0, 0xa91,
     0xa91, 0xc31, 0xa91, 0x9c1, 0x1a0, 0xc31, 0x1a0, 0x681, 0x410, 0x821, 0x9c1, 0x410, 0xd0,  0xc31, 0x340, 0x8f1,
     0x270, 0x270, 0x410, 0x681, 0x0,   0x4e0, 0x340, 0x5b0, 0x340, 0x410, 0xc31, 0x821, 0xb61, 0x4e0, 0x270, 0x681,
     0x5b0, 0xd0,  0x410, 0x5b0, 0xc31, 0x821, 0x340, 0x8f1, 0xb61, 0x9c1, 0xb61, 0x270, 0x9c1, 0x0,   0x751, 0x821},
    {0xd0,  0x681, 0xbc9, 0x751, 0xd0,  0xd0,  0x3a8, 0x8f1, 0x1a0, 0xc31, 0x410, 0xaf9, 0xbc9, 0x2d8, 0xb61, 0x681,
     0x5b0, 0x5b0, 0x208, 0x410, 0x618, 0x751, 0x681, 0x959, 0x68,  0xa29, 0xd0,  0x68,  0x4e0, 0x208, 0xa91, 0x208,
     0x959, 0x5b0, 0x1a0, 0xa91, 0x2d8, 0x3a8, 0x138, 0xa91, 0x4e0, 0x138, 0x548, 0x9c1, 0x0,   0xa29, 0x7b9, 0x138,
     0xa91, 0x889, 0xa29, 0x138, 0x208, 0x2d8, 0x548, 0xd0,  0x751, 0x478, 0x340, 0x2d8, 0xc99, 0x751, 0x3a8, 0x138,
     0x751, 0x821, 0x751, 0x618, 0x138, 0xc99, 0x270, 0x889, 0x8f1, 0x959, 0x8f1, 0x618, 0x618, 0x959, 0xc99, 0x2d8,
     0x681, 0x68,  0xa29, 0xa91, 0xa29, 0x7b9, 0x889, 0x340, 0x270, 0xa91, 0x2d8, 0x0,   0x6e9, 0x7b9, 0xaf9, 0x618,
     0x959, 0xc31, 0x340, 0xa29, 0x6e9, 0xaf9, 0x270, 0x68,  0x618, 0x478, 0x3a8, 0x270, 0x138, 0x68,  0x0,   0x68,
     0x9c1, 0x1a0, 0x618, 0x478, 0x8f1, 0xa91, 0x208, 0x68,  0x2d8, 0x5b0, 0xa91, 0x9c1, 0x2d8, 0xb61, 0xc31, 0x2d8,
     0x0,   0xbc9, 0xaf9, 0xbc9, 0x3a8, 0xc31, 0x959, 0xbc9, 0x270, 0x208, 0x8f1, 0x618, 0x68,  0x681, 0x0,   0x8f1,
     0x959, 0x7b9, 0x3a8, 0x68,  0x0,   0x959, 0xaf9, 0x138, 0xc31, 0x548, 0x68,  0xd0,  0x410, 0x9c1, 0x751, 0x681,
     0xc99, 0x4e0, 0x9c1, 0x3a8, 0xc31, 0x1a0, 0xc99, 0x889, 0x0,   0x478, 0x1a0, 0x270, 0x681, 0x340, 0xd0,  0x9c1,
     0x478, 0x340, 0x478, 0x9c1, 0x4e0, 0x478, 0xd0,  0x478, 0x5b0, 0x340, 0x9c1, 0x340, 0x889, 0x9c1, 0xbc9, 0x821,
     0x0,   0x1a0, 0x0,   0x681, 0x889, 0x6e9, 0x3a8, 0xa91, 0x68,  0xa29, 0x270, 0x478, 0xc31, 0x4e0, 0x548, 0xaf9,
     0xaf9, 0xc99, 0xa29, 0xa29, 0x208, 0xc31, 0x1a0, 0x681, 0x478, 0x7b9, 0x9c1, 0x410, 0x138, 0xc31, 0x340, 0x889,
     0x270, 0x270, 0x478, 0x681, 0x0,   0x4e0, 0x340, 0x5b0, 0x340, 0x410, 0xc31, 0x889, 0xb61, 0x478, 0x270, 0x618,
     0x5b0, 0xd0,  0x478, 0x5b0, 0xc99, 0x821, 0x3a8, 0x959, 0xb61, 0x9c1, 0xaf9, 0x270, 0x9c1, 0x0,   0x6e9, 0x821},
    {0xd0,  0x64c, 0xb95, 0x71d, 0xd0,  0x104, 0x3a8, 0x8f1, 0x1d4, 0xbfd, 0x444, 0xac5, 0xbfd, 0x2d8, 0xb61, 0x6b5,
     0x57c, 0x5b0, 0x1d4, 0x3dc, 0x64c, 0x751, 0x64c, 0x925, 0x34,  0x9f5, 0xd0,  0x68,  0x4ac, 0x1d4, 0xa5d, 0x23c,
     0x959, 0x5b0, 0x1d4, 0xa91, 0x2a4, 0x3a8, 0x138, 0xac5, 0x4ac, 0x16c, 0x57c, 0x98d, 0x34,  0xa29, 0x7ed, 0x138,
     0xac5, 0x855, 0xa29, 0x138, 0x208, 0x30c, 0x57c, 0x104, 0x751, 0x478, 0x30c, 0x30c, 0xccd, 0x785, 0x3a8, 0x138,
     0x751, 0x821, 0x785, 0x618, 0x104, 0xc65, 0x270, 0x8bd, 0x8bd, 0x959, 0x925, 0x64c, 0x618, 0x98d, 0xccd, 0x30c,
     0x681, 0x9c,  0xa29, 0xa5d, 0xa29, 0x7b9, 0x889, 0x340, 0x23c, 0xa91, 0x30c, 0x0,   0x71d, 0x785, 0xac5, 0x618,
     0x959, 0xc31, 0x374, 0xa5d, 0x6b5, 0xaf9, 0x270, 0x9c,  0x618, 0x478, 0x3a8, 0x270, 0x138, 0x68,  0x0,   0x68,
     0x9c1, 0x1a0, 0x618, 0x478, 0x925, 0xa5d, 0x208, 0x34,  0x2d8, 0x57c, 0xa91, 0x9c1, 0x30c, 0xb95, 0xc65, 0x2d8,
     0x0,   0xbc9, 0xaf9, 0xbc9, 0x374, 0xc31, 0x959, 0xbc9, 0x23c, 0x208, 0x8f1, 0x618, 0x68,  0x64c, 0xccd, 0x8bd,
     0x98d, 0x7b9, 0x3dc, 0x68,  0xccd, 0x959, 0xaf9, 0x104, 0xc65, 0x514, 0x68,  0x9c,  0x444, 0x9c1, 0x751, 0x6b5,
     0xc99, 0x4e0, 0x9f5, 0x3a8, 0xc31, 0x1d4, 0xccd, 0x889, 0x0,   0x478, 0x1a0, 0x270, 0x64c, 0x340, 0x9c,  0x9c1,
     0x444, 0x374, 0x478, 0x9c1, 0x4e0, 0x478, 0x9c,  0x478, 0x57c, 0x30c, 0x9c1, 0x30c, 0x889, 0x98d, 0xbc9, 0x821,
     0x34,  0x16c, 0xccd, 0x681, 0x889, 0x6b5, 0x3a8, 0xa91, 0x68,  0x9f5, 0x23c, 0x478, 0xc65, 0x4ac, 0x548, 0xac5,
     0xac5, 0xc65, 0xa5d, 0xa29, 0x208, 0xbfd, 0x16c, 0x6b5, 0x444, 0x7ed, 0x9c1, 0x3dc, 0x138, 0xc31, 0x374, 0x8bd,
     0x270, 0x270, 0x478, 0x681, 0x0,   0x4e0, 0x374, 0x5b0, 0x374, 0x410, 0xbfd, 0x855, 0xb2d, 0x4ac, 0x23c, 0x64c,
     0x5b0, 0xd0,  0x478, 0x5e4, 0xc65, 0x821, 0x3a8, 0x959, 0xb61, 0x9f5, 0xb2d, 0x270, 0x9c1, 0xccd, 0x71d, 0x821},
    {0xb6,  0x64c, 0xb95, 0x737, 0xd0,  0x104, 0x3c2, 0x90b, 0x1ba, 0xc17, 0x42a, 0xadf, 0xbe3, 0x2d8, 0xb61, 0x6b5,
     0x596, 0x5ca, 0x1ee, 0x3f6, 0x64c, 0x76b, 0x64c, 0x925, 0x34,  0xa0f, 0xb6,  0x68,  0x4c6, 0x1ee, 0xa77, 0x23c,
     0x973, 0x5b0, 0x1ba, 0xa77, 0x2be, 0x3c2, 0x11e, 0xac5, 0x4ac, 0x16c, 0x57c, 0x9a7, 0x34,  0xa29, 0x7d3, 0x152,
     0xaab, 0x855, 0xa29, 0x11e, 0x1ee, 0x30c, 0x57c, 0xea,  0x751, 0x478, 0x30c, 0x2f2, 0xcb3, 0x76b, 0x38e, 0x152,
     0x737, 0x821, 0x76b, 0x632, 0x11e, 0xc7f, 0x270, 0x8a3, 0x8bd, 0x959, 0x925, 0x632, 0x618, 0x973, 0xcb3, 0x2f2,
     0x69b, 0x9c,  0xa29, 0xa77, 0xa29, 0x79f, 0x889, 0x340, 0x23c, 0xa77, 0x30c, 0x0,   0x703, 0x79f, 0xac5, 0x5fe,
     0x959, 0xc31, 0x374, 0xa5d, 0x6b5, 0xb13, 0x270, 0x9c,  0x632, 0x45e, 0x3c2, 0x270, 0x138, 0x68,  0x0,   0x68,
     0x9a7, 0x186, 0x618, 0x478, 0x925, 0xa5d, 0x208, 0x4e,  0x2be, 0x596, 0xa77, 0x9a7, 0x2f2, 0xb7b, 0xc65, 0x2be,
     0x0,   0xbe3, 0xaf9, 0xbaf, 0x374, 0xc31, 0x959, 0xbc9, 0x23c, 0x208, 0x8f1, 0x618, 0x4e,  0x64c, 0xccd, 0x8d7,
     0x973, 0x7b9, 0x3c2, 0x4e,  0xce7, 0x959, 0xaf9, 0x104, 0xc4b, 0x514, 0x68,  0xb6,  0x444, 0x9c1, 0x751, 0x6b5,
     0xc99, 0x4fa, 0x9f5, 0x3a8, 0xc17, 0x1ba, 0xcb3, 0x8a3, 0x0,   0x492, 0x1a0, 0x270, 0x666, 0x35a, 0x9c,  0x9a7,
     0x45e, 0x374, 0x478, 0x9c1, 0x4e0, 0x478, 0x9c,  0x492, 0x596, 0x326, 0x9a7, 0x30c, 0x889, 0x98d, 0xbc9, 0x821,
     0x1a,  0x186, 0xccd, 0x666, 0x889, 0x6b5, 0x3a8, 0xaab, 0x68,  0x9f5, 0x23c, 0x492, 0xc4b, 0x4ac, 0x548, 0xadf,
     0xac5, 0xc7f, 0xa43, 0xa29, 0x1ee, 0xbfd, 0x186, 0x69b, 0x444, 0x7ed, 0x9a7, 0x3f6, 0x138, 0xc31, 0x374, 0x8a3,
     0x256, 0x270, 0x478, 0x69b, 0x0,   0x4e0, 0x374, 0x5b0, 0x374, 0x42a, 0xbfd, 0x86f, 0xb2d, 0x492, 0x23c, 0x632,
     0x5ca, 0xd0,  0x478, 0x5e4, 0xc7f, 0x821, 0x38e, 0x93f, 0xb47, 0x9f5, 0xb13, 0x270, 0x9c1, 0xce7, 0x703, 0x807},
    {0xc3,  0x659, 0xb95, 0x72a, 0xc3,  0x104, 0x3b5, 0x90b, 0x1c7, 0xc17, 0x42a, 0xadf, 0xbf0, 0x2d8, 0xb54, 0x6a8,
     0x589, 0x5bd, 0x1ee, 0x3e9, 0x64c, 0x75e, 0x659, 0x925, 0x34,  0xa0f, 0xc3,  0x5b,  0x4b9, 0x1ee, 0xa6a, 0x23c,
     0x973, 0x5bd, 0x1ba, 0xa77, 0x2be, 0x3c2, 0x12b, 0xab8, 0x4ac, 0x16c, 0x57c, 0x99a, 0x34,  0xa1c, 0x7d3, 0x152,
     0xab8, 0x855, 0xa36, 0x11e, 0x1fb, 0x30c, 0x57c, 0xea,  0x751, 0x46b, 0x319, 0x2f2, 0xcc0, 0x76b, 0x38e, 0x145,
     0x737, 0x821, 0x778, 0x632, 0x111, 0xc72, 0x263, 0x8a3, 0x8bd, 0x959, 0x925, 0x63f, 0x60b, 0x980, 0xcb3, 0x2ff,
     0x68e, 0x8f,  0xa29, 0xa77, 0xa1c, 0x7ac, 0x889, 0x340, 0x249, 0xa77, 0x30c, 0x0,   0x710, 0x79f, 0xad2, 0x5fe,
     0x94c, 0xc31, 0x374, 0xa50, 0x6b5, 0xb06, 0x270, 0x9c,  0x632, 0x45e, 0x3c2, 0x270, 0x12b, 0x75,  0x0,   0x75,
     0x9a7, 0x193, 0x60b, 0x478, 0x925, 0xa5d, 0x208, 0x4e,  0x2cb, 0x596, 0xa77, 0x9b4, 0x2ff, 0xb7b, 0xc65, 0x2be,
     0xd,   0xbe3, 0xaf9, 0xbaf, 0x374, 0xc24, 0x94c, 0xbc9, 0x23c, 0x215, 0x8f1, 0x625, 0x5b,  0x659, 0xcda, 0x8ca,
     0x973, 0x7ac, 0x3c2, 0x5b,  0xce7, 0x94c, 0xb06, 0x104, 0xc58, 0x514, 0x68,  0xa9,  0x437, 0x9c1, 0x751, 0x6b5,
     0xca6, 0x4fa, 0x9f5, 0x39b, 0xc24, 0x1ba, 0xcc0, 0x8a3, 0xd,   0x492, 0x193, 0x270, 0x666, 0x35a, 0xa9,  0x9a7,
     0x451, 0x367, 0x478, 0x9c1, 0x4e0, 0x478, 0xa9,  0x492, 0x596, 0x319, 0x9b4, 0x30c, 0x889, 0x99a, 0xbc9, 0x814,
     0x27,  0x186, 0xccd, 0x673, 0x889, 0x6b5, 0x3b5, 0xa9e, 0x75,  0x9f5, 0x23c, 0x492, 0xc4b, 0x4ac, 0x548, 0xad2,
     0xac5, 0xc7f, 0xa50, 0xa29, 0x1ee, 0xbfd, 0x186, 0x69b, 0x444, 0x7ed, 0x9b4, 0x3e9, 0x138, 0xc24, 0x374, 0x8b0,
     0x256, 0x270, 0x478, 0x68e, 0x0,   0x4e0, 0x367, 0x5a3, 0x367, 0x42a, 0xbfd, 0x862, 0xb3a, 0x492, 0x23c, 0x63f,
     0x5ca, 0xdd,  0x46b, 0x5d7, 0xc72, 0x82e, 0x38e, 0x94c, 0xb54, 0x9e8, 0xb13, 0x263, 0x9ce, 0xcda, 0x703, 0x814},
    {0xbd,  0x653, 0xb95, 0x72a, 0xca,  0x104, 0x3bc, 0x904, 0x1c7, 0xc10, 0x42a, 0xad8, 0xbf0, 0x2df, 0xb5a, 0x6ae,
     0x590, 0x5c4, 0x1ee, 0x3e9, 0x64c, 0x764, 0x659, 0x925, 0x3b,  0xa0f, 0xbd,  0x5b,  0x4b9, 0x1e8, 0xa6a, 0x23c,
     0x973, 0x5bd, 0x1c1, 0xa7d, 0x2be, 0x3c2, 0x12b, 0xabe, 0x4b3, 0x16c, 0x57c, 0x99a, 0x2e,  0xa1c, 0x7d9, 0x152,
     0xab8, 0x855, 0xa36, 0x11e, 0x1f5, 0x306, 0x576, 0xea,  0x751, 0x46b, 0x313, 0x2f9, 0xcc0, 0x771, 0x395, 0x145,
     0x73d, 0x81a, 0x771, 0x632, 0x118, 0xc72, 0x263, 0x8a3, 0x8bd, 0x95f, 0x925, 0x63f, 0x60b, 0x979, 0xcb3, 0x2ff,
     0x68e, 0x8f,  0xa22, 0xa70, 0xa22, 0x7ac, 0x88f, 0x340, 0x249, 0xa77, 0x30c, 0x0,   0x710, 0x798, 0xacb, 0x605,
     0x94c, 0xc31, 0x36e, 0xa56, 0x6b5, 0xb06, 0x270, 0x96,  0x62c, 0x465, 0x3bc, 0x26a, 0x132, 0x75,  0x0,   0x75,
     0x9ad, 0x193, 0x612, 0x478, 0x925, 0xa5d, 0x208, 0x4e,  0x2cb, 0x590, 0xa77, 0x9ad, 0x2f9, 0xb81, 0xc65, 0x2c5,
     0x7,   0xbdc, 0xaff, 0xbb5, 0x37b, 0xc2a, 0x94c, 0xbc9, 0x243, 0x215, 0x8f7, 0x625, 0x5b,  0x659, 0xcd3, 0x8d0,
     0x973, 0x7ac, 0x3c9, 0x5b,  0xce0, 0x952, 0xaff, 0x104, 0xc58, 0x514, 0x62,  0xa9,  0x43e, 0x9ba, 0x74a, 0x6ae,
     0xc9f, 0x4f4, 0x9f5, 0x39b, 0xc1d, 0x1c1, 0xcb9, 0x89c, 0xd,   0x492, 0x193, 0x270, 0x666, 0x35a, 0xa9,  0x9a7,
     0x458, 0x36e, 0x472, 0x9c7, 0x4da, 0x478, 0xa3,  0x48c, 0x596, 0x320, 0x9b4, 0x30c, 0x889, 0x993, 0xbcf, 0x814,
     0x21,  0x180, 0xccd, 0x673, 0x889, 0x6bb, 0x3b5, 0xa9e, 0x75,  0x9fb, 0x23c, 0x48c, 0xc51, 0x4ac, 0x54f, 0xad2,
     0xacb, 0xc7f, 0xa49, 0xa29, 0x1f5, 0xbfd, 0x186, 0x6a1, 0x444, 0x7e6, 0x9b4, 0x3f0, 0x138, 0xc24, 0x374, 0x8b0,
     0x256, 0x26a, 0x472, 0x68e, 0x0,   0x4e0, 0x367, 0x5a3, 0x36e, 0x42a, 0xc03, 0x862, 0xb33, 0x499, 0x23c, 0x639,
     0x5ca, 0xd7,  0x46b, 0x5d7, 0xc72, 0x827, 0x38e, 0x945, 0xb4d, 0x9e8, 0xb19, 0x263, 0x9ce, 0xce0, 0x703, 0x814},
    {0xc0,  0x656, 0xb98, 0x72d, 0xca,  0x104, 0x3b9, 0x907, 0x1c4, 0xc10, 0x42e, 0xad8, 0xbed, 0x2db, 0xb5a, 0x6ae,
     0x58d, 0x5c1, 0x1eb, 0x3ed, 0x64c, 0x764, 0x659, 0x925, 0x3b,  0xa0c, 0xc0,  0x5b,  0x4bd, 0x1e8, 0xa6d, 0x239,
     0x96f, 0x5bd, 0x1bd, 0xa7d, 0x2be, 0x3c2, 0x12b, 0xabb, 0x4b3, 0x16c, 0x579, 0x99a, 0x2e,  0xa1c, 0x7d6, 0x14f,
     0xab8, 0x858, 0xa33, 0x121, 0x1f5, 0x309, 0x579, 0xea,  0x754, 0x46b, 0x316, 0x2f5, 0xcc0, 0x76e, 0x392, 0x145,
     0x73a, 0x81a, 0x774, 0x632, 0x118, 0xc72, 0x266, 0x8a3, 0x8c0, 0x95f, 0x925, 0x63c, 0x60b, 0x97c, 0xcb3, 0x2fc,
     0x68e, 0x92,  0xa22, 0xa74, 0xa22, 0x7a8, 0x88f, 0x344, 0x246, 0xa7a, 0x309, 0x3,   0x710, 0x79b, 0xacf, 0x605,
     0x94c, 0xc34, 0x371, 0xa53, 0x6b8, 0xb06, 0x270, 0x99,  0x62c, 0x462, 0x3bf, 0x26a, 0x132, 0x72,  0x0,   0x72,
     0x9ad, 0x190, 0x60f, 0x478, 0x921, 0xa5d, 0x208, 0x4b,  0x2c8, 0x593, 0xa7a, 0x9ad, 0x2f9, 0xb81, 0xc62, 0x2c1,
     0x7,   0xbe0, 0xaff, 0xbb5, 0x378, 0xc27, 0x94c, 0xbc9, 0x23f, 0x215, 0x8f4, 0x625, 0x5b,  0x659, 0xcd3, 0x8d0,
     0x973, 0x7af, 0x3c6, 0x5b,  0xce4, 0x94f, 0xb03, 0x104, 0xc58, 0x514, 0x62,  0xac,  0x43b, 0x9ba, 0x74a, 0x6ae,
     0xca3, 0x4f4, 0x9f5, 0x39f, 0xc21, 0x1bd, 0xcb9, 0x89f, 0xd,   0x492, 0x196, 0x273, 0x666, 0x35a, 0xa9,  0x9aa,
     0x455, 0x36b, 0x475, 0x9c4, 0x4da, 0x47c, 0xa3,  0x48c, 0x593, 0x31c, 0x9b0, 0x30f, 0x885, 0x993, 0xbcc, 0x817,
     0x24,  0x180, 0xcd0, 0x673, 0x889, 0x6bb, 0x3b5, 0xa9e, 0x72,  0x9fb, 0x23c, 0x48f, 0xc51, 0x4b0, 0x54c, 0xad5,
     0xacb, 0xc7c, 0xa4d, 0xa26, 0x1f1, 0xbfd, 0x186, 0x6a1, 0x448, 0x7e6, 0x9b0, 0x3ed, 0x138, 0xc24, 0x374, 0x8ac,
     0x259, 0x26a, 0x472, 0x68e, 0x3,   0x4e0, 0x36b, 0x5a3, 0x36b, 0x427, 0xc03, 0x862, 0xb37, 0x499, 0x23c, 0x639,
     0x5ca, 0xd7,  0x46f, 0x5d7, 0xc72, 0x82a, 0x392, 0x948, 0xb51, 0x9e8, 0xb19, 0x266, 0x9cb, 0xce0, 0x703, 0x810},
    {0xc0,  0x656, 0xb97, 0x72d, 0xc8,  0x102, 0x3b9, 0x906, 0x1c4, 0xc10, 0x42c, 0xad8, 0xbed, 0x2dd, 0xb59, 0x6ac,
     0x58d, 0x5c1, 0x1eb, 0x3ed, 0x64b, 0x764, 0x658, 0x925, 0x39,  0xa0d, 0xc0,  0x5d,  0x4bd, 0x1e9, 0xa6d, 0x23b,
     0x96f, 0x5bc, 0x1bf, 0xa7d, 0x2bd, 0x3c1, 0x129, 0xabd, 0x4b1, 0x16a, 0x57b, 0x99a, 0x2e,  0xa1d, 0x7d6, 0x150,
     0xab6, 0x856, 0xa34, 0x120, 0x1f5, 0x307, 0x579, 0xec,  0x752, 0x46b, 0x314, 0x2f5, 0xcbe, 0x76f, 0x392, 0x147,
     0x73b, 0x81c, 0x774, 0x631, 0x118, 0xc74, 0x265, 0x8a4, 0x8be, 0x95f, 0x925, 0x63c, 0x60b, 0x97c, 0xcb3, 0x2fe,
     0x68f, 0x92,  0xa22, 0xa72, 0xa22, 0x7a8, 0x88f, 0x344, 0x248, 0xa7a, 0x30b, 0x2,   0x710, 0x79b, 0xacd, 0x603,
     0x94c, 0xc34, 0x371, 0xa55, 0x6b8, 0xb06, 0x270, 0x99,  0x62e, 0x462, 0x3bf, 0x26b, 0x132, 0x73,  0x2,   0x72,
     0x9ad, 0x191, 0x60f, 0x478, 0x921, 0xa5e, 0x208, 0x4c,  0x2ca, 0x592, 0xa7a, 0x9ad, 0x2f9, 0xb81, 0xc63, 0x2c1,
     0x8,   0xbe0, 0xafe, 0xbb4, 0x379, 0xc29, 0x94d, 0xbcb, 0x241, 0x214, 0x8f4, 0x625, 0x5b,  0x659, 0xcd3, 0x8d0,
     0x974, 0x7ad, 0x3c6, 0x59,  0xce4, 0x951, 0xb01, 0x106, 0xc56, 0x516, 0x63,  0xac,  0x43b, 0x9ba, 0x74c, 0x6b0,
     0xca3, 0x4f4, 0x9f5, 0x39d, 0xc21, 0x1bf, 0xcb9, 0x89f, 0xb,   0x491, 0x196, 0x272, 0x666, 0x359, 0xa7,  0x9a8,
     0x455, 0x36c, 0x473, 0x9c6, 0x4da, 0x47c, 0xa4,  0x48d, 0x595, 0x31c, 0x9b2, 0x30f, 0x887, 0x993, 0xbce, 0x815,
     0x24,  0x180, 0xcd0, 0x673, 0x88a, 0x6b9, 0x3b4, 0xa9e, 0x73,  0x9fa, 0x23e, 0x48f, 0xc50, 0x4b0, 0x54c, 0xad3,
     0xacb, 0xc7d, 0xa4b, 0xa27, 0x1f3, 0xbff, 0x184, 0x6a1, 0x446, 0x7e8, 0x9b0, 0x3ed, 0x138, 0xc24, 0x374, 0x8ae,
     0x258, 0x26a, 0x472, 0x68e, 0x3,   0x4df, 0x369, 0x5a5, 0x36b, 0x429, 0xc02, 0x863, 0xb37, 0x497, 0x23e, 0x639,
     0x5ca, 0xd7,  0x46f, 0x5d7, 0xc72, 0x82a, 0x390, 0x947, 0xb4f, 0x9e9, 0xb19, 0x266, 0x9cc, 0xcdf, 0x704, 0x810},
    {0x9b,  0x526, 0x96b, 0x5d4, 0xa3,  0xd3,  0x307, 0x755, 0x16f, 0x9cf, 0x364, 0x8d1, 0x9b1, 0x254, 0x939, 0x56d,
     0x483, 0x4ad, 0x18f, 0x330, 0x51d, 0x602, 0x528, 0x76f, 0x2e,  0x82b, 0x9b,  0x4b,  0x3d9, 0x18d, 0x879, 0x1cf,
     0x7ac, 0x4a9, 0x16b, 0x886, 0x23a, 0x30d, 0xf2,  0x8b9, 0x3d1, 0x127, 0x473, 0x7ce, 0x25,  0x838, 0x65e, 0x111,
     0x8b5, 0x6c7, 0x84b, 0xea,  0x197, 0x277, 0x473, 0xbf,  0x5f4, 0x398, 0x281, 0x268, 0xa5c, 0x60b, 0x2e7, 0x109,
     0x5e0, 0x696, 0x60f, 0x508, 0xe3,  0xa1e, 0x1f2, 0x706, 0x71c, 0x79e, 0x76e, 0x512, 0x4ea, 0x7b5, 0xa52, 0x26e,
     0x555, 0x77,  0x83d, 0x87d, 0x83c, 0x63a, 0x6f5, 0x2a7, 0x1da, 0x884, 0x278, 0x2,   0x5bd, 0x62e, 0x8c7, 0x4e4,
     0x78e, 0x9eb, 0x2cc, 0x865, 0x575, 0x8f6, 0x1fb, 0x7c,  0x505, 0x390, 0x30b, 0x1f7, 0xf8,  0x5d,  0x1,   0x5d,
     0x7dd, 0x146, 0x4ed, 0x3a2, 0x76c, 0x86d, 0x1a7, 0x3e,  0x243, 0x487, 0x884, 0x7de, 0x26a, 0x95a, 0xa11, 0x23e,
     0x7,   0x9a6, 0x8ef, 0x983, 0x2d3, 0x9e1, 0x78f, 0x995, 0x1d5, 0x1b0, 0x747, 0x4fe, 0x4a,  0x529, 0xa6d, 0x72a,
     0x7af, 0x63e, 0x311, 0x49,  0xa79, 0x792, 0x8f2, 0xd4,  0xa07, 0x422, 0x50,  0x8c,  0x370, 0x7e9, 0x5ed, 0x56f,
     0xa44, 0x407, 0x817, 0x2f0, 0x9db, 0x16b, 0xa58, 0x702, 0xa,   0x3b6, 0x14a, 0x1fd, 0x533, 0x2b9, 0x89,  0x7da,
     0x385, 0x2c8, 0x39f, 0x7f1, 0x3f1, 0x3a4, 0x85,  0x3b3, 0x489, 0x288, 0x7e1, 0x27c, 0x6ee, 0x7c9, 0x997, 0x692,
     0x1c,  0x138, 0xa6a, 0x53e, 0x6f1, 0x578, 0x302, 0x8a1, 0x5d,  0x81c, 0x1d2, 0x3b4, 0xa02, 0x3ce, 0x44e, 0x8cd,
     0x8c6, 0xa26, 0x85e, 0x840, 0x196, 0x9bf, 0x13c, 0x563, 0x379, 0x66c, 0x7e0, 0x331, 0xfe,  0x9de, 0x2ce, 0x70e,
     0x1e8, 0x1f6, 0x39d, 0x554, 0x2,   0x3f5, 0x2c6, 0x496, 0x2c7, 0x361, 0x9c2, 0x6d1, 0x91d, 0x3bc, 0x1d2, 0x50f,
     0x4b4, 0xaf,  0x39a, 0x4c0, 0xa1d, 0x6a3, 0x2e5, 0x78a, 0x931, 0x80e, 0x905, 0x1f3, 0x7f6, 0xa76, 0x5b4, 0x68e},
  };

  run_all_tests(reference_input, reference_result, reference_decompressed, std::make_index_sequence<11>{});
}

template <const uint8_t k, const uint8_t eta1>
__launch_bounds__(128) __global__ void pke_keygen(
  const uint8_t* d, // random 32B
  uint8_t* ek,      // encap key (size 384 * k + 32 bytes)
  uint8_t* dk,      // decap key (size 384 * k bytes)
  Zq* A             // A (size 256 * k * k bytes)
)
{
  // update the pointers according to the batch index
  d += blockIdx.x * 32;
  ek += blockIdx.x * (384 * k + 32);
  dk += blockIdx.x * (384 * k);
  A += blockIdx.x * 256 * k * k;
  pke::keygen<k, eta1>(d, ek, dk, PolyMatrix<256, k, k, Zq>(A));
}

TEST_F(KyberTest, PkeKeygen512)
{
  const std::string d = "1af17a664e3fa8e419b8ba05c2a173169df76162a5a286e0c405b460d478f7ef";
  const std::string ek_pke =
    "3e62893ae983c681a800874203448b2f749ee2ea3a91b17e5e5227b681634757c6cdb7c3d02802a4038729129eb367805d6664a8863992eba4"
    "4deb40e588b725fc71fbe91fbb27195e1b59c66649c1684636dc0526b4421b522c991a8b5666509983c814893e732b7ef0d6c8a031784df167"
    "7fea006f4147325c72ca8a8d1d7356e5c55160e74cc67614db3c6c0351770e5347ddb8a04b997521eb69b4a27f8c4c6cf60b0b419a5a4de907"
    "06283ad8b37f4c132a94bbb4fbb1b17b610a1992215d1c2b47806738c029d5ec2e45e3a5b016875654292d1730bbfc6d8dd2730fa2050c2514"
    "c63b8758a9732cb119c0a08c9d9b68a5e0193cfc1ef3f4517c3b673bbb2b61f4676a0b45b8796477799ec77438c593cbb01265e4a69e82d655"
    "bf1833c4fa45b32ba86a90a9f3e949c942b591c8124e194d07fcabb883a873074772d30ee1ea769c3b8d991615a77757ed005d712944063034"
    "3a8612fa473206910fedda4b1ec4702df91056333f3ae54594745c179b23c16237fb8197837402e64910be5608c37b6ca35c6bdf438c7c2577"
    "1afb5a9209a8f4e48588e4bd6749a4f8b895d30420de2b7916b94b95b0cbd640237c13c0124b407f407ec29b00bea953bc4a01233576f795ca"
    "6629ce07237c03020f02270f43a73a30ea40d90a719a9302490995b5dc57cb7c6f6f0ab8c58129da095b86a51daeaa2776f683b6ea52b0d989"
    "20ec81ff08b904ca1a85e8960c499046f7a50d072a30a999ee0c732da1a30b775bbd485b1b018fcaa2634e9c13f969389bf71817f7af0bb929"
    "75bb4426e4ccf508488867a24c1970543a59b4db00f5033b51a207fd616551c8416261c2d6392e085c1eb8d35a2db35126f1be6077991b7357"
    "5ca60b2aa928ac1627bcdc1bba40631e9152df95717c8a440741c2d6e79a010c1a650baaab86528ee23c001bcac0951fe8eacdeb242dc26b21"
    "111925c5d0377472b1498859d01c2be69b8f33d756dc0473c63b5b4eb7abb78794cb4272cab4ba65003415d33e7f12a91cd43785a22138c5a6"
    "39e6b922248aa685b0894a7de3e63489f82e7e398aba5622f4f829ec6e912ca1729d2ec0c233409baa3d6b105eb7f862e97ad86a367a8b798c"
    "325c";
  const std::string dk_pke =
    "153524580b8e23a49f4de93c36fabc0f6011e4c4b80cda7c5ec897d40321721710ec21157c51b7ffe87741322f11704358c52dbc1c6e66bcb9"
    "5f1156e0987572db795508226f3a10451761269b76e8d875bfbb02aac42499e5b0b14965ae6975373cb1a3c82ae5c495c7e15bc314bdcc0471"
    "cd15a148e179d508483da76c122b5612bc34849664e25a085e6409ced245286851a103386b0163851a0ef7f525aa54b9fa6a34d128c9b29672"
    "8328c4cf4032ab9602e9669e6132bfcc7bbb345718e9d209b6990fc9d30c5fb5ca2f960e74e1b2e9c420762908ebf17b69477db2f1230d233c"
    "8893178946bb12a49bcef01b65e33d4519a3d4d86402451794645f497748e491b288419fc3d724e9713a7e6b87ea6989d080ceb40ab79e39cc"
    "a9ec8b3253427721c110e42e81149c8329cc854b6ac7f7a7ddd01c8e5784b40bb91e687a7fa5692ca9777b267cab0c858e80c3b7471689b583"
    "f6537475dba22434bb5ab253e2bba40284a4d2c72380f229d210386a56cfa2175ee62183b288182e36c413c02af5f914a0c332aad6b69ecb49"
    "da623a8bb87fdac666d8525ff642c3a8c3756b9271217c48a829a3e9ea499c43bd1cc6ca2900cf1ef0587da84b6ca54928f795d6702609a7b0"
    "34b0c248888b34a7501676770f7b079e45c87bc4b24fa5a8e2921f45d4aeecd8711824a021022452ebcf19e577b54842511c41a8642dbba0c5"
    "906940440236850204ed467d5dc89934028119200af1c5c58d4cc6cd40c7b68437a55c1ed0a32af8dc356c4468449512e5c5ac48c61260c8bf"
    "eb82a749ab1fdcc1bc249b4b02b660707b004d659872f23231452a69e67910dc41404185a0f372484b4cf3d0adf0760b4d7cbddb777a813766"
    "6ae76892827d3002371233a9d76295ca268b7fd71c09abb39f10835866c9d95506060270546690112cc7bee479eab8bbee7cc20397a4cbdb83"
    "34560476a70228f2aea6a956a9e1a312643556e89af53b3b2cba53d197762c6b9f0ef5c03252ca1735683c126f70793f3dd8918052a50ddb6b"
    "a06c36bb08373e625bccac8173529405d9a8edc2a268dc7fa643b7";

  // Convert hex string to byte array
  std::vector<uint8_t> d_bytes;
  for (size_t i = 0; i < d.length(); i += 2) {
    d_bytes.push_back((uint8_t)std::stoi(d.substr(i, 2), nullptr, 16));
  }

  const uint8_t KYBER_K = 2;
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 384 * KYBER_K;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  cudaMalloc((void**)&d_d, 32);
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  cudaMemcpy(d_d, d_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Launch kernel
  // pke_keygen512<<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  pke_keygen<2, 3><<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(ek_size);
  std::vector<uint8_t> h_dk(dk_size);
  cudaMemcpy(h_ek.data(), d_ek, ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, dk_size, cudaMemcpyDeviceToHost);

  // Convert to hex string for comparison
  std::string result_ek = voidPtrToHexString((const std::byte*)h_ek.data(), ek_size);
  std::string result_dk = voidPtrToHexString((const std::byte*)h_dk.data(), dk_size);

  ASSERT_EQ(result_dk, dk_pke) << "Decapsulation key mismatch";
  ASSERT_EQ(result_ek, ek_pke) << "Encapsulation key mismatch";
  // ASSERT_EQ(result_dk, dk_pke) << "Decapsulation key mismatch";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

template <const uint8_t k = 2>
__global__ void ntt_inplace_test_kernel(Zq* s_e)
{
  // NTT on s and e
  for (int i = 0; i < k * 2; i++) {
    ntt_inplace(Poly<256, Zq>(s_e + 256 * i));
  }
}

TEST_F(KyberTest, NttInplace)
{
  uint16_t s[2][256] = {
    {0,    1,    3328, 3328, 0,    3327, 3328, 0,    1,    0,    1,    3328, 0,    1,    1,    1,    3327, 1,    2,
     3328, 0,    3327, 3327, 0,    3328, 1,    3327, 1,    3328, 1,    3328, 3327, 0,    3327, 0,    3328, 1,    0,
     0,    3327, 3327, 0,    3328, 1,    1,    1,    3328, 1,    0,    0,    3328, 0,    3328, 3328, 2,    0,    2,
     1,    3327, 3328, 0,    0,    1,    2,    0,    2,    0,    1,    3327, 3328, 1,    0,    1,    3328, 0,    0,
     0,    1,    3327, 1,    0,    0,    3328, 0,    2,    1,    3327, 1,    1,    1,    1,    3328, 0,    3328, 0,
     0,    1,    1,    1,    1,    0,    3328, 2,    3327, 1,    3328, 0,    1,    3328, 3327, 3328, 3328, 2,    0,
     1,    3327, 0,    0,    0,    1,    1,    1,    0,    3328, 0,    3328, 3328, 0,    0,    0,    3326, 0,    2,
     3327, 3327, 3327, 0,    1,    3328, 1,    3328, 3328, 0,    2,    3327, 0,    2,    3328, 0,    3328, 0,    3328,
     0,    3328, 2,    3327, 2,    0,    1,    0,    1,    3327, 0,    0,    3328, 2,    0,    1,    3327, 3326, 3328,
     0,    3326, 3328, 2,    0,    0,    0,    3328, 3327, 1,    0,    1,    0,    3328, 3328, 3328, 3328, 2,    3327,
     2,    2,    1,    0,    0,    0,    0,    3327, 3327, 0,    0,    0,    0,    0,    3328, 3328, 1,    0,    3328,
     0,    1,    0,    1,    2,    0,    3327, 0,    3328, 1,    2,    0,    3327, 0,    3327, 3328, 0,    0,    0,
     0,    3327, 1,    3328, 3328, 1,    0,    0,    3328, 3327, 0,    3327, 3328, 3327, 1,    3327, 3327, 0,    1,
     3328, 1,    3328, 3328, 1,    3327, 2,    0,    1},
    {3328, 0,    0,    1,    0,    2,    2,    1,    3328, 0,    1,    0,    3328, 0, 2,    3328, 1,    3327, 1,
     3328, 1,    1,    0,    3328, 2,    0,    0,    0,    1,    3327, 1,    2,    2, 0,    0,    0,    1,    3327,
     1,    3327, 0,    2,    1,    3328, 2,    0,    3327, 2,    3328, 0,    0,    2, 0,    3328, 2,    0,    0,
     0,    0,    0,    0,    3327, 0,    1,    0,    1,    2,    0,    0,    3328, 2, 1,    3327, 1,    3328, 3327,
     0,    3328, 3327, 0,    0,    3328, 2,    1,    0,    0,    1,    3328, 0,    0, 3327, 0,    3328, 1,    2,
     3328, 3,    3328, 3327, 1,    0,    3328, 0,    3328, 3328, 0,    0,    0,    0, 0,    1,    2,    0,    1,
     0,    0,    1,    0,    0,    1,    0,    3328, 2,    2,    3328, 2,    0,    1, 0,    3328, 0,    0,    0,
     0,    3327, 3328, 3327, 3328, 2,    3327, 3328, 3,    0,    0,    3326, 3328, 1, 1,    0,    0,    0,    3328,
     0,    1,    3,    0,    3326, 1,    3328, 1,    3328, 1,    0,    3328, 3327, 1, 1,    0,    0,    3328, 0,
     1,    3328, 3328, 3328, 1,    3328, 1,    0,    0,    2,    3328, 0,    3328, 0, 0,    2,    3328, 0,    1,
     3328, 1,    0,    0,    0,    1,    0,    1,    2,    0,    1,    3328, 0,    0, 1,    1,    0,    1,    3328,
     1,    0,    1,    1,    0,    2,    0,    1,    1,    3328, 3328, 3,    0,    0, 1,    3327, 1,    0,    3328,
     3328, 3328, 0,    3327, 1,    1,    1,    2,    1,    0,    0,    3327, 3328, 1, 3327, 2,    3327, 3328, 0,
     3328, 3328, 1,    0,    3328, 2,    0,    2,    3327}};

  uint16_t e[2][256] = {
    {0,    3327, 1,    0,    1,    1,    0,    3328, 2,    0,    0,    0,    3328, 0,    3328, 3328, 0,    3328, 3328,
     0,    0,    0,    1,    3328, 0,    3328, 0,    0,    2,    1,    3327, 0,    0,    1,    3327, 3,    0,    0,
     1,    0,    3327, 3328, 0,    0,    3328, 1,    3328, 3,    0,    3328, 1,    3328, 1,    2,    1,    3328, 0,
     0,    3327, 1,    3328, 0,    3326, 0,    3327, 0,    3,    3328, 3328, 2,    3328, 3328, 1,    1,    3327, 3328,
     3,    3328, 1,    3328, 0,    3328, 3328, 3327, 3328, 2,    0,    0,    1,    1,    3,    3328, 0,    0,    0,
     1,    3328, 2,    3328, 3328, 0,    0,    1,    2,    0,    3328, 0,    1,    3328, 3328, 3327, 0,    1,    3328,
     3328, 0,    3328, 1,    1,    0,    0,    0,    2,    3328, 3328, 3327, 1,    1,    3327, 3328, 0,    3327, 3328,
     2,    3328, 0,    3328, 0,    1,    3328, 0,    3328, 3328, 3328, 1,    3328, 0,    3328, 0,    3,    1,    3328,
     3328, 0,    1,    3328, 0,    0,    1,    1,    1,    2,    0,    3328, 0,    3327, 0,    0,    3328, 0,    1,
     3328, 1,    0,    1,    3327, 1,    0,    0,    2,    0,    3327, 0,    0,    3328, 3328, 1,    2,    0,    2,
     3327, 0,    0,    1,    0,    3328, 1,    0,    3327, 3328, 3328, 1,    3328, 1,    1,    3328, 0,    1,    3328,
     0,    3328, 0,    0,    3327, 3328, 0,    3327, 3328, 3328, 3328, 1,    1,    3328, 1,    1,    1,    1,    1,
     3327, 0,    0,    0,    1,    1,    1,    3,    0,    0,    3328, 0,    3328, 1,    1,    3328, 2,    3328, 2,
     0,    1,    0,    3327, 0,    0,    1,    0,    0},
    {0,    3328, 0,    3328, 3328, 3328, 0,    3327, 0,    3327, 0,    0,    3328, 1,    3326, 3327, 1,    3328, 0,
     3328, 1,    0,    0,    3327, 3327, 3328, 1,    3327, 3328, 3327, 0,    2,    0,    3327, 3328, 0,    2,    3327,
     1,    0,    0,    3327, 3328, 0,    0,    3328, 2,    0,    2,    0,    0,    2,    2,    0,    1,    3328, 3327,
     2,    3328, 1,    0,    1,    0,    3328, 0,    1,    0,    3328, 2,    2,    1,    1,    0,    1,    0,    1,
     0,    3328, 1,    0,    0,    1,    0,    2,    3327, 3328, 0,    0,    3327, 3327, 0,    3328, 3328, 3328, 3,
     0,    0,    1,    0,    3328, 3327, 0,    2,    0,    3328, 0,    3328, 3328, 3327, 0,    2,    1,    1,    0,
     2,    0,    0,    3328, 3328, 0,    3327, 0,    3328, 1,    3328, 0,    2,    3328, 1,    1,    3328, 0,    1,
     0,    0,    3328, 0,    1,    1,    3328, 0,    0,    1,    0,    2,    0,    2,    1,    0,    0,    3328, 0,
     0,    3327, 0,    0,    1,    3327, 0,    1,    2,    0,    2,    0,    0,    1,    0,    3328, 1,    3328, 0,
     1,    0,    3328, 3328, 3327, 0,    1,    1,    0,    0,    3328, 3327, 0,    3328, 0,    0,    3327, 0,    1,
     3328, 3328, 1,    1,    2,    2,    0,    3,    0,    0,    1,    3327, 3328, 0,    0,    0,    3326, 3327, 2,
     3328, 3328, 3328, 2,    1,    0,    2,    3328, 3328, 1,    0,    3328, 2,    2,    3328, 3328, 1,    3328, 3328,
     0,    0,    1,    0,    2,    1,    2,    0,    3328, 1,    3328, 0,    1,    0,    0,    0,    3328, 3328, 2,
     1,    3328, 0,    1,    1,    0,    1,    1,    1}};

  uint16_t s_hat[2][256] = {
    {1301, 579,  2904, 2272, 1059, 2554, 2381, 974,  2614, 3023, 15,   278,  1252, 2956, 2572, 1997, 2142, 2428, 980,
     528,  1906, 257,  492,  338,  380,  2933, 2303, 1918, 577,  755,  17,   1079, 1368, 732,  3260, 1761, 3174, 2971,
     351,  1377, 2272, 1881, 2930, 1949, 2133, 544,  2671, 259,  1861, 1553, 2854, 1897, 2280, 1885, 3007, 43,   1194,
     588,  1433, 2830, 2481, 1620, 2478, 1878, 3127, 2835, 2211, 684,  1253, 2396, 455,  1470, 1219, 3025, 1228, 1808,
     1485, 2577, 328,  1950, 2261, 1152, 1853, 1738, 2834, 1378, 3090, 843,  1668, 1609, 2786, 133,  1118, 150,  718,
     1117, 2088, 1302, 929,  896,  363,  1584, 2693, 225,  1527, 607,  1194, 2965, 2810, 838,  2257, 3218, 1714, 1833,
     2179, 3138, 207,  804,  1707, 41,   1769, 2534, 609,  3059, 3020, 2999, 1844, 389,  745,  157,  2486, 249,  969,
     205,  1375, 3243, 1583, 233,  372,  2862, 1257, 524,  2422, 130,  491,  1983, 1897, 2004, 434,  575,  781,  962,
     904,  377,  1673, 2996, 1042, 2490, 206,  447,  869,  990,  2373, 2609, 2260, 1613, 1282, 372,  1172, 1526, 1865,
     1159, 484,  2857, 392,  2548, 1987, 589,  489,  935,  2942, 2166, 2538, 2198, 208,  3304, 2740, 2928, 2462, 3267,
     3241, 2238, 818,  1061, 375,  3090, 1040, 750,  1153, 2497, 2435, 3266, 2949, 1700, 1991, 2687, 221,  461,  1934,
     2117, 2996, 2960, 2078, 1958, 1407, 1690, 2348, 1914, 1659, 1986, 3243, 2128, 142,  3128, 1975, 356,  1417, 2107,
     1014, 1861, 2933, 2605, 1060, 2995, 602,  1339, 3042, 2635, 1026, 2632, 2002, 572,  640,  671,  210,  897,  1642,
     3317, 1954, 1505, 486,  2098, 2226, 392,  1582, 3139},
    {19,   684,  2549, 335,  928,  812,  1706, 2925, 2974, 1180, 730,  934,  2187, 2043, 1754, 1644, 728,  1525, 758,
     3124, 936,  1884, 619,  1817, 3105, 1159, 2472, 2610, 2793, 1182, 924,  3028, 1564, 3244, 41,   3312, 30,   1423,
     2173, 1210, 1388, 1178, 1832, 2399, 214,  615,  1801, 2826, 52,   3115, 2120, 2232, 1844, 1290, 1558, 1911, 2831,
     119,  1438, 3204, 1147, 2860, 1359, 2698, 738,  505,  1093, 2797, 2284, 1821, 1048, 2562, 545,  576,  2898, 3326,
     1305, 1918, 2229, 1060, 3153, 1041, 1192, 726,  187,  3162, 2448, 1030, 580,  864,  645,  64,   1773, 2004, 2141,
     2460, 564,  2064, 25,   162,  1521, 3164, 3213, 3172, 205,  3188, 1206, 888,  3237, 485,  976,  682,  3320, 861,
     1132, 1668, 1348, 297,  1509, 2764, 1608, 300,  2144, 3068, 747,  2680, 2889, 506,  476,  3020, 2852, 1209, 1538,
     1547, 2928, 7,    1357, 2438, 626,  815,  1329, 676,  1641, 1950, 3088, 1053, 320,  2132, 928,  1839, 2888, 1220,
     243,  2781, 1776, 183,  3149, 3031, 2011, 1959, 1921, 1635, 1898, 1678, 658,  2008, 560,  880,  786,  2707, 727,
     2390, 1738, 2226, 1919, 461,  2825, 2874, 159,  2097, 1624, 3222, 1497, 101,  518,  1792, 1620, 2310, 3089, 3186,
     1214, 1950, 2282, 3003, 3310, 3111, 1795, 2633, 3019, 2109, 1588, 69,   1910, 42,   552,  2799, 2470, 1386, 425,
     2622, 1042, 854,  2134, 2478, 3061, 947,  2604, 1339, 2001, 1897, 2860, 2550, 1294, 3087, 562,  3237, 1303, 1667,
     572,  1777, 2416, 1015, 2109, 2333, 640,  2645, 2829, 1725, 3232, 870,  2235, 880,  574,  1462, 3276, 2074, 627,
     2373, 2309, 2701, 749,  2604, 3176, 2045, 934,  2932}};

  uint16_t e_hat[2][256] = {
    {1511, 1159, 863,  2864, 374,  3257, 2384, 2810, 2551, 145,  602,  1662, 1875, 2174, 3106, 1849, 626,  2589, 209,
     352,  480,  280,  1273, 855,  780,  3328, 401,  2816, 913,  2453, 581,  3208, 925,  840,  2459, 3037, 1758, 101,
     1862, 311,  1283, 2949, 1309, 3230, 1208, 2119, 316,  2841, 3207, 1072, 2730, 952,  2134, 3287, 2875, 1949, 1551,
     1424, 1881, 1976, 1543, 1839, 1567, 115,  1782, 938,  1290, 1118, 45,   599,  2025, 2448, 2028, 3252, 1927, 2820,
     274,  1114, 1263, 2395, 3303, 492,  1680, 1431, 1470, 2551, 3218, 1240, 240,  3037, 534,  3324, 686,  247,  457,
     1192, 1293, 2504, 443,  838,  1552, 239,  2914, 2657, 190,  3105, 2120, 824,  2033, 925,  886,  1544, 2448, 2718,
     3048, 697,  112,  3272, 3024, 2063, 3281, 6,    3195, 331,  1439, 1214, 935,  1281, 1300, 446,  2946, 2717, 2348,
     1823, 2040, 619,  1063, 2728, 1094, 295,  2922, 351,  598,  2206, 1886, 1521, 1244, 2091, 1811, 940,  2607, 3145,
     930,  913,  1506, 3028, 2423, 2629, 2094, 2173, 87,   384,  1280, 972,  1361, 183,  2291, 589,  1780, 2918, 2768,
     2329, 622,  869,  3181, 1363, 3212, 1393, 1083, 560,  1155, 2505, 1440, 1394, 1985, 1257, 2584, 3069, 2580, 520,
     402,  11,   1290, 3269, 1790, 906,  2719, 2522, 2403, 754,  2657, 3262, 2317, 3185, 2980, 1890, 161,  1667, 3228,
     2210, 468,  2457, 169,  1806, 1755, 3040, 1432, 351,  847,  3228, 3057, 1458, 2765, 1168, 2211, 2954, 1301, 203,
     3317, 410,  1348, 2283, 1859, 1985, 1736, 972,  2838, 676,  1767, 656,  2248, 1636, 1849, 3165, 2393, 2276, 3153,
     1781, 2572, 570,  899,  2308, 1044, 2384, 917,  1826},
    {995,  507,  1886, 690,  3077, 1845, 837,  2799, 758,  1412, 1230, 3010, 615,  1562, 2614, 1443, 364,  1668, 1601,
     274,  1474, 2034, 3055, 35,   1778, 2561, 1545, 2081, 2640, 527,  2489, 1887, 2129, 3291, 3199, 3050, 1448, 2918,
     1260, 911,  1452, 1696, 1053, 200,  2324, 2358, 956,  1624, 1941, 138,  106,  3251, 1804, 1053, 2434, 3097, 2693,
     1057, 1206, 2161, 1027, 2376, 2058, 3164, 664,  2628, 2312, 174,  3256, 2029, 1974, 1182, 691,  2621, 210,  1952,
     1799, 1841, 126,  2709, 939,  225,  1825, 2763, 2141, 1308, 924,  1062, 3287, 361,  3216, 329,  2722, 1031, 508,
     1365, 649,  1631, 358,  3175, 1232, 2092, 1469, 2371, 1461, 2232, 1656, 2112, 2271, 2891, 1938, 2727, 1411, 2623,
     478,  1571, 1025, 3100, 3099, 2893, 1793, 1667, 1075, 1321, 280,  3225, 3161, 1359, 1302, 594,  1828, 1214, 296,
     3253, 1355, 1996, 3042, 2852, 2710, 87,   331,  1381, 2091, 872,  2724, 191,  2801, 1914, 1652, 923,  1866, 2371,
     290,  982,  1527, 3057, 2830, 460,  3019, 1116, 2844, 2904, 1160, 2825, 393,  635,  2266, 1201, 2277, 1363, 1449,
     2489, 3081, 1181, 2290, 352,  1132, 1931, 910,  1568, 1773, 818,  1233, 3179, 1612, 668,  141,  2765, 2288, 1854,
     2680, 227,  3315, 2214, 242,  1443, 557,  1613, 3069, 2565, 1431, 143,  1498, 3141, 2546, 3285, 167,  920,  2845,
     560,  17,   3207, 1113, 3002, 2836, 1196, 772,  225,  1421, 2506, 211,  1820, 686,  2294, 2648, 2426, 899,  153,
     2109, 3173, 1931, 663,  2264, 1508, 791,  1516, 1288, 524,  2024, 1217, 2142, 2966, 2101, 2230, 2486, 923,  2658,
     1512, 1718, 1433, 2349, 67,   2093, 959,  2967, 1709}};

  const uint8_t KYBER_K = 2;

  // Allocate device memory for s_e array (which will hold both s and e)
  Zq* d_s_e;
  cudaMalloc((void**)&d_s_e, 2 * KYBER_K * 256 * sizeof(Zq));

  // Copy s and e to device memory in the correct format
  Zq* h_s_e = new Zq[2 * KYBER_K * 256];
  for (int i = 0; i < KYBER_K; i++) {
    for (int j = 0; j < 256; j++) {
      h_s_e[i * 256 + j] = Zq(s[i][j]);
      h_s_e[(KYBER_K + i) * 256 + j] = Zq(e[i][j]);
    }
  }
  cudaMemcpy(d_s_e, h_s_e, 2 * KYBER_K * 256 * sizeof(Zq), cudaMemcpyHostToDevice);

  // Run the NTT inplace kernel
  ntt_inplace_test_kernel<<<1, 128>>>(d_s_e);

  // Copy results back to host
  cudaMemcpy(h_s_e, d_s_e, 2 * KYBER_K * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < KYBER_K; i++) {
    for (int j = 0; j < 256; j++) {
      ASSERT_EQ(h_s_e[i * 256 + j].raw(), s_hat[i][j]) << "Mismatch in s_hat at position [" << i << "][" << j << "]";
      ASSERT_EQ(h_s_e[(KYBER_K + i) * 256 + j].raw(), e_hat[i][j])
        << "Mismatch in e_hat at position [" << i << "][" << j << "]";
    }
  }

  // Cleanup
  delete[] h_s_e;
  cudaFree(d_s_e);
}

TEST_F(KyberTest, PkeKeygen512Batch)
{
  const uint64_t batch = 2048 * 4;
  // const uint64_t batch = 2048;
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const uint8_t KYBER_K = 2;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 384 * KYBER_K;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate memory for batch
  cudaMalloc((void**)&d_d, 32 * batch);
  cudaMalloc((void**)&d_ek, ek_size * batch);
  cudaMalloc((void**)&d_dk, dk_size * batch);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq) * batch);

  // Generate random input data
  uint8_t* h_d = new uint8_t[32 * batch];
  for (uint64_t i = 0; i < 32 * batch; i++) {
    h_d[i] = rand() % 256;
  }
  cudaMemcpy(d_d, h_d, 32 * batch, cudaMemcpyHostToDevice);

  // Launch kernel for batch
  pke_keygen<KYBER_K, 3><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Cleanup
  delete[] h_d;
  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, PkeKeygen768)
{
  const std::string d = "3c53596d802b9371d9893e3822a35af22d3b872a2561f10b7072aced348c0d9d";
  const std::string ek_pke =
    "9fe43f4b4b9b1e264e9099732b7ccad1f054bb1657045c639a789105581d3fd87a988b776e400a14f229aedb497d831c196b47f3bbcae16c20"
    "e5279959b97012e0bb0ca0b8d5e14029e1587a5108df81a4413378c8064e945234e4966a68ca2e36f38e2f6215c54b99459a1e7f22c2d68655"
    "a1e14ca3ebbf13b369c2e64eb6f72afaf5603e851703165cb0e92fa5e736fb451021aa73fef342e953a62afbc837294ac9f1347ee7870cbb11"
    "abe3aa0eeca7d7750b99848bb537a4e4886d464941670bc1ae0408f548c7e5cb14c52942cb775da9d2268616cf1d75b886d1a632a66993233f"
    "46c34cab9a7d3342ce4027c097d67789b491837711f29ba5d5134acd975d4903416508aecc0a719fa580c73492f9643d5f82369224a312c829"
    "986389ea3acde1e6346e512ce34884ec9150cc8603dce502b2c1018beab6944a380f9b86bed1a56fb393a57069954835f7e74a128059e0201f"
    "c99b3c4dd37a32303c075665b867bacd3c0612a822196269fdb4925bdc9603476ab636710d943449e53b9142c3620c476b2c67c6e40c18d039"
    "94b114a7224ab194b70f0ab6cd00c6291240f816bdd1d85355da158595194221c99a11632cd07d07e3afef677e692b512041839b4b872432bf"
    "ef09745bd63b5a78011a73740a7a12424843ce793d58c02e005b137711afc7bcbee396a602c4936b2883bed153c4821e64f0930eeab888f94b"
    "36508b1680bd2f06b979466597770ebb65b30ff04050845ce6254c342a3bf7324212c92997e65560498ef8f259de860ffe86ce4164b7e2bb52"
    "2a9796ea679097f3c733878656370a16a786ac91703330c4976b15c322c6f07ab04d81ad3e7ac8c29314a4bc8a5f526a811068494297c4fca3"
    "55981e6396c37760284661626df244adb6287c63b80eea7f9e566939c51f1eaa979f8a7aa7abced7ec4643eb5497d05f83f212ab80327e7894"
    "5a9c310fc13104b91a9018074276b21ad80fc552983be858b0086fc2a667d502772c523397225fbbccae385a8391bc1843260bb71960d7e3c1"
    "958c844e4142c3f9747e149a1afa5517b65244379879c87a07c38a41058fea24695a491d866a47980a6f834a16ec5a6ca0d671397854896bb9"
    "65b724133c649e1c4b6eeb6a1f1673a6b91ff60c171db796af2690434cb0722bbc86081868b137cc0a42cdaabf62d4c62fecc5bbb06605865d"
    "f0a49e0f9b2f4685baa622749b88ae89099e5bb58910e25cffa307f94999539bc319c667e9c6632e3a20bab617cdb49fd4c085f48032531968"
    "108c4535eb8c2cda3bfdbc2b175440fee4ad42ca72f6f56b7ef69f99b39476d5a200044eb2693c1a98b36c9b74ff4bbacbc5b9c96067b36198"
    "acf23afc446ee1051f051938bf5cc4ea50c6de51cd8be6a8e690896512c74d0175454c9e84ea2d5edaabb3963900fb2c4eb96725840d33712d"
    "fb440635a39435335483992114e924ced72cae9684ac70aa959cc06e74a3b2bc4d5ccabd73269c9dbacf44fb8f11dc38deb953563c089a550d"
    "f2448e28856ac70272774cada6c311c30bc7d6d133ac8796dec2cb39572506c02a175bc10a916f76c4204278a975dc6376b728b8119c38c01c"
    "17b562db87348f891f4aca4d1bce73abcca37928795df969cf2418e4dd97f4b141719aa0c07ad19d987e23cc";
  const std::string dk_pke =
    "bee3788e04492c635d31b48ad5a5cfb056a241f25cc2126ec0d537c7a86690143b4b8013682736339293d6f3759c0b467ba6a6e0077f5082bc"
    "56056255b182fdd05557ab86eb995052ccb41b185b6b34aae6d39955306f0360b0d314a6dd8b8bc3d74f48012d2f12c19b985fd7795c94924b"
    "cdf7b7e01c8c20376f461aa0c72772c814206af55e5fb54a41454758a1971b26816251074c939ec6388f798c6f6cdcb44a233f260ab7a4322c"
    "b0882843b34ef9481204d69e188868a6f24fe8091bb7419904b76c8cfb0c0019860609b02f51a3a7fb35671a758cc652cad94fefacceacf796"
    "94a69def389063fa1082490928e860b68a3bd1a6b18bc7215ad64021a024eeba39d67657e4e56cbcc4061782c839585b74738cfe1c519203a3"
    "7c5769de30bb76377ce179ba8ce961845a36c3cc12e546c453b41f41513e1f369a7e9490cf53b4b57b6411fb0f234503ef8626e55082e3023d"
    "3e1c0602795cded8ac7f69b62594a1dda92581d61878ca86cbd26e7ce455d879a735772c290b9ff695cf987382cd7c72bab2be8450ba729029"
    "11dc596645cbbe48c38cd8c5529c29bedb83039599607bb5605ca2f504247cfa870bea89326b71ec22ce791abafe1831b66c0794dac75c1556"
    "fe2324a4253d0db94a72384117a6293db8b1d08719b636566f15b44f03b29f5313db535a8fb363933a3618c65a27c9a9152a9c47a49e974528"
    "6e40b2261b855f18cb7948af70042358e44cbaa24ddb5234d67b1363d39e0a60ae021153733c5c5758b18a26bdf56aaeff9157769a7e0de52b"
    "7a0046aae314abb2c0a00c10982a7fa7fb8004ba32042ab8d0a60b5c4b00ee59c2246620e02c325a1c5ecb1acdb00a2f0835ad2016a6860132"
    "ddda8da5bab3a7d62c1ab21494d06fa7bb54313b4afc7731cb9858d7e0bd0ad164779b0723838f86007bba8b9d19c00d48937b1402003504aa"
    "bba98e360a440ffb3c71db68ff9c4536c9395a6aa9c7c1119d628d37f809bcfca202611fe3b59db307a3f3d5abfd1cb89dc6bd53e497986c60"
    "e0998a2eec496d23049090b22e39067e831eab5b433241a359c52586c0c18ea4c8a7f2a2a18831032184002c6e5a07cd86f21ad85b5e88246d"
    "51fb73769a14d711927eea1bc253868f040569d12e17781b5c5367b0e743f1587d0eb2341e12c90a0c37dca86449010221fbc63c1ab78e946b"
    "83d1aca8051616eba79b46bf7a74826acc145563cf75fc9e2364a0b299c3d6607c5423158bd075d6e55e0f2c3cd35a2141425aa6615c0e740d"
    "9a883b6aa040f7231b2393664f78c6959cbb9bb436163798120cbcc611936b0271fd4156e2bcc85404aa9f133fc679aec288b84cd9ac774c4d"
    "d14b9ed233b8c700bf4c91682eda5e6fe9817dea122ad123760556b5c450f6827afa216278a62d3f6a91ade19e7b90367530a65baba7196566"
    "7fc161af92a719d773ad993752855abaf0625323621a68a50e4c9a0059c75548618ea3c6ce3c6e27e62230646d08645ec849155548715f60c1"
    "44232e78c3c66edb2382e187896056180670b36078f7b6385e25b674fc855c6163cfd9b666700805e6270718701695b8dc080e8bd7b566a3b7"
    "4fb04ca378a872550c06706e";

  // Convert hex string to byte array
  std::vector<uint8_t> d_bytes;
  for (size_t i = 0; i < d.length(); i += 2) {
    d_bytes.push_back((uint8_t)std::stoi(d.substr(i, 2), nullptr, 16));
  }

  const uint8_t KYBER_K = 3;
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 384 * KYBER_K;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  cudaMalloc((void**)&d_d, 32);
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  cudaMemcpy(d_d, d_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Launch kernel d_A);
  pke_keygen<KYBER_K, 2><<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(ek_size);
  std::vector<uint8_t> h_dk(dk_size);
  cudaMemcpy(h_ek.data(), d_ek, ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, dk_size, cudaMemcpyDeviceToHost);

  // Convert to hex string for comparison
  std::string result_ek = voidPtrToHexString((const std::byte*)h_ek.data(), ek_size);
  std::string result_dk = voidPtrToHexString((const std::byte*)h_dk.data(), dk_size);

  ASSERT_EQ(result_dk, dk_pke) << "Decapsulation key mismatch";
  ASSERT_EQ(result_ek, ek_pke) << "Encapsulation key mismatch";
  // ASSERT_EQ(result_dk, dk_pke) << "Decapsulation key mismatch";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, PkeKeygen1024)
{
  const std::string d = "3c53596d802b9371d9893e3822a35af22d3b872a2561f10b7072aced348c0d9d";
  const std::string ek_pke =
    "2532443ab10a01707841212d1be31da291b9b5903e09a3822e9a4bda239567f3349b451c11a345f4166005cb662baa2ca4e86cb52668007d86"
    "e9f64951c591dae578673a0b485469e19b0f4e45c07bb272bd584a39777fdf7c7051b188c820c46ba75e4bb785c5e17aceeaafa9d33aaa776f"
    "adaa7a27e2a8bb9b16f2414d11703fc4ec0c7a5046cd06b766469d3526775c473fcd51b7d304cf4dab78100b05b0fa16e7c603f441ca6dc13e"
    "6504329d9b8e371b93da33b0fce7b8cb62985f12a893f876b2550fc4bb456053770e39244b57c448d36b1fd36b83902eb29c68ce05331d0810"
    "4fd92b2913b3f95c1a95a59947d44aba0b4830a4c112cc08b414bc9503543637666f945ff390523fa5600087a5efa31ab3c65ac6f5c96be546"
    "bcec803204c4de646e6a8a634bfb9ec57b7dfd160917fb8daaaa21a7872dcec569a3f30fea65758d125f21b63680c28280d8b0dcb937230aa2"
    "a32b5bdf816d52bc1d40f84fde351c460317a439795fa791b7ca3703c966d69b60f142c4706640da532b9d36966e88026660456ff3678db645"
    "b5035b06334c6c0533860b27d0f04deed38322c17054636971c24b65c7729bc9c4a01ba2aef93bf58aa2835105f9a23fb87425229b6c5fc8ab"
    "e49a9f402085c823cd5b08cbc1b85a52d1444c2a3603a096b1182b239c106fc196f77500c3416ecd66643d0bc927d2a24e310437735c3719a2"
    "ff7199c791270001151f387bc4264306c57a62934395239d9419556705caf635c2b1b02df21a2a8ac37841722cdaec3a7b854432c56e0f9131"
    "a83c688d2a2a8e8453a4901a8351b6da01b57728b4799b32ca6c93f3f9bf266a3ed38859bad45ebb71914c1a4a95494be8f4bb50f92a380358"
    "c272785bebcbcd3573614bb05e80cbeb1614dcba50470b7f577a9dff7b1ce1d31b33618e8eb47957d6a1cb03c318841153c00477055a40c18e"
    "174014f5a22142689654f018531ca5106976acb6cebd5008d9a1a1731482d9b1c313c493ac5100ae35040df15cd89b25ab071cb31115efa001"
    "2d270605c53dd445c1713a5cdcac486cdac8cc27b48d1630492b9a44c28c11e2ac8034820d885b3f4ab15564c4c0175e57caa7f054b771c6c2"
    "6fb3088815438ff64f7a9382f119adc3958bed636b0b8671e7b73c33b1bb2618518a24bd3382cdda867e1f461fa8f2c978b77bc3747d2dc163"
    "1b127cc54206b2ec039915913678c670b20c38d39832e98c749cb1eddb275e2340bcdb2899425eb485aae6694631e8758d5c15ac4ba1658b3b"
    "e87173f25a69532812f49240365579741801f677657d757c7d68a53d856c537553e8cb68670212196a8c08f507489ab0ba51a1d3e78896f88b"
    "657744c44c7e7d24baf45c070d3818be7ba6d4f944ec4babecd120d8a5b7275bc3b405cf81a3028dbba6a43cc027bc05d3771f3849a66c4745"
    "158b9aa5b8c87e323b0cc353c0f8ab53013869e6a2103832d062095804b19a657ea26b50af59236bd20dcd496c1ba13a1472396c21956d29c5"
    "5e36c5665a403a2869b6c19dd8879240c1cc56d1b7513998f4c72046e657a04c900573ad8e8b50f53619b6664bda45b552ecbecf83a630933e"
    "367130ad9c3be3541b2580ce096a041ca179e6e682c3c478bc659a7ad00cd72621a3b6c10cba353ada16bd814380971759147148b1950ce660"
    "2e0b275ec712819c1494f65cb71084c522130ab97426936d225377d9119a8f765b4fd84f7c1bbce583c55e9bbef697ba74492aaa7cba51217d"
    "bd1670a8833e5979484961528eb48f1b225af9a725a22581249b0dc969a9c4f48966238eade848ee7a22e0457aa8872af1eb605a842f48442a"
    "41200311fc4f99b21f36b621641112a9accfc1ac384243cb91e2aeb966aa49fbccb3c61de3e626ab5b4507d956ccd9a02db9ac2a910e1bd631"
    "b496505314bbf7897fa9b77bfa0b5fd04544b1190399c24adf769354088109e0cddad16a6427581be09f676c36d372107cab8d3cd57e5a0264"
    "e6220d0a0081fa936220a829802144204730d16917b8e9c555ca024ec9adc9e80d8b8c6a77a85e52351dcddccae3f8b0b0060e4bd43cde1659"
    "e0a20e0259bdece18e7b609065927ee49243a66c4683448e1baa0db2908acee1c768f692092802579183be0ca16e0293054a7980a337c5c583"
    "6e796dc4bdda58b0a9d2a880107ceae767af37319fc4ef83e38b5d78c8";
  const std::string dk_pke =
    "f26b04b7c86b95b9516709182bda98e0c3a84324a29b252d7e6764e300c52c42a9ecd277ad31bc79219ec19a9d2217442a41c14d9836234b36"
    "bf2ccf1be68b54c042827467f7ca7a464a434b078f74575bf75459eb8668ffcc08b4743f7ca0cad64a5606a4b7e1f87999663e0ef54019282b"
    "3e1019e71c5413f3a8ac1628f8e0c13885c064e4a328866fc2799f597315e28c2ebf69b3062cc12142affdd2a9b516662c046548c7316fb1b9"
    "f3527c69152296780de115015da61de0b24bf2599da7c86ec0b4264f5c5e8c758b7fdaa55fa2acaaeb2d973333d0ec14fb506250084c00fdab"
    "1d004acc229fc2097482b7ca34339cbda8c6a2d43480b65c13877de2d3290081a8c1a02d32fa226de0af590731b7d2a8c8513108f4b4054169"
    "50f1a3b54543ceba0b0bb34ceee2532f86bc8a9a079b9179542347f62040886890057a52686c86664c97a318263bb92171951d41e81e184271"
    "86b10488db6d1ea3950686b83f226221068dfda795e4db1798186b91c33629825483e18bd7b1343c0430d843a2ec123c0278693bd9ab264133"
    "36ab314de176c084a7a0d5c4488661a62240d611790118506dc692a1b891c1b7901502ad110c8fd2d5b59db44f225b32dd505ff841689c6813"
    "1268525e2458a6e8c234dc370d6948b347b5bf5a8a1925a38f880cc427a2116a3fb7998db4f56f60095d10f41c5627295c07a29e2239f0d737"
    "6f5b6a1f6a35f9c59a2999384a4a91abb449d41694de1367c5a659aaf2c1f5062e5257c15c7194cfa6106a278ae756b6efb6116f553c2422c3"
    "c551c32938ae905c951b6114c7279d9a687eac2a69d55b125af40d6901c56d4366db2b6986e65df23b3f59a272d4b158e7a32493e084948c74"
    "fd55befec643bfdc79839547b7095e51c25178a74ab9f73be01096add7687cb22f4190b725548707641e71bb6ce194c97febab90e2764a179c"
    "86612b79ec0fd59731b20601edf002d4105ea097034754438159b011e4b4373c7d78078a6236017a1b48a375ab332205c88bb3f242a41f738a"
    "9dea3bcb0378f18443d71751e464326803502f705a8895b00f211caa911088770d5daa96743547a4d34707f233154a2ea43023a9bb1470855a"
    "98656ee373893af82cbd51661cc6b90b3c5552552cb7296c35c795cf541c5b2a8cd14202368aad10471c3d66cf3da2479f4243e3c09e988000"
    "d1f49c53e9947547b8ebdb2bce482aab00c30b6677eb910284dcb6c42520039abdc95042e9f20bcc3967c6d352aeab85121132be8087052b6a"
    "c7600ea9e947b611c1b493b2abeccaca4bb4da4940835b5862ea0d892a2f0209c9b13312b9ac11ebd2535b2a23450b0a0da2bcc8385f6bd276"
    "e1b6b2ca987f4c55aa71c9ac5b463d7943cf0b528c1d917f72f8917d136065331f2edc73fc52a23e819bd39a282b75afddf6436ef4b7223447"
    "24074a8261ac9b760d33d9bd3d43bbd389c7d87860eeab053ae511f1cb736fe40783b24a0288126121ad977c42a4452489fabfe8f5042f2612"
    "657a44031c3bfcac13661a03bcaa1503a0295da11fc26507233104b9d77fb62284a3b5807c0ba3af808f43cba7c4c6200c245f85551049c555"
    "8ed35127b554cff8b5a3e429d41b0beaf8caa0b4a96e013619e68a7aba17c1fc4a6040042944b7efa46858f57a7e12cd52872fcb08cb026b6f"
    "13d24d9a7434bc2ab65dda4aa7415cb148344afa1dc418a20efbbee8b833a249638da0072cc328b6930522913339473dd7694810c60e120a98"
    "a31145e170ba88a4caa7c3afed026348998847395ff208a934fbb5a4fc5e898c3ec7ccc92bf84c5e2a82ae022c34954b75239520a7c8ed9321"
    "36da8b37433cd85a117667bc4a65808d400d6f16489d07c6be6ba1179387274950f923a4a7b6ce9d4142e254b2b007541b75ca40bc9b979a48"
    "88e9a647795af7981f89b42c057108e72090d2f5b4cd1a93f6474f6f229566548f16a639224cb724ac086cdc81c3f659fd788fd31c30b73b20"
    "f2e483607651f6bcc17471cc6467a23e12c6d657943bf0a37bbb0a7d8180af5b825b2a0bf2527d55a8879a979e7a011898303d683ace8a8a28"
    "9ec89f4e42c6419b335f93010f5c6f127a745e722e835bcdb7779f96149d02e50282f4286a0a6243e5b611dc7badd78523893da66b38";

  // Convert hex string to byte array
  std::vector<uint8_t> d_bytes;
  for (size_t i = 0; i < d.length(); i += 2) {
    d_bytes.push_back((uint8_t)std::stoi(d.substr(i, 2), nullptr, 16));
  }

  const uint8_t KYBER_K = 4;
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 384 * KYBER_K;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  cudaMalloc((void**)&d_d, 32);
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  cudaMemcpy(d_d, d_bytes.data(), 32, cudaMemcpyHostToDevice);

  // Launch kernel
  pke_keygen<KYBER_K, 2><<<1, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Copy results back
  std::vector<uint8_t> h_ek(ek_size);
  std::vector<uint8_t> h_dk(dk_size);
  cudaMemcpy(h_ek.data(), d_ek, ek_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dk.data(), d_dk, dk_size, cudaMemcpyDeviceToHost);

  // Convert to hex string for comparison
  std::string result_ek = voidPtrToHexString((const std::byte*)h_ek.data(), ek_size);
  std::string result_dk = voidPtrToHexString((const std::byte*)h_dk.data(), dk_size);

  ASSERT_EQ(result_dk, dk_pke) << "Decapsulation key mismatch";
  ASSERT_EQ(result_ek, ek_pke) << "Encapsulation key mismatch";
  // ASSERT_EQ(result_dk, dk_pke) << "Decapsulation key mismatch";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, SampleA768)
{
  const std::string rho = "47e7c222944b9c20fec896c9c81743d020e61513bdb1d30e33ec3cf6bfc49e6a";

  uint16_t A[3][3][256] = {
    {{306,  3048, 601,  661,  1773, 2482, 2457, 1598, 3097, 2848, 1425, 946,  2007, 3160, 551,  2251, 2917, 1494, 1331,
      129,  415,  3162, 1311, 2889, 2179, 2708, 1477, 302,  2778, 1646, 1171, 1759, 2639, 1789, 2767, 2919, 3095, 524,
      988,  2828, 2504, 1946, 3320, 715,  309,  3017, 758,  2814, 611,  2821, 1707, 2624, 729,  2051, 218,  2348, 1096,
      2216, 1871, 2847, 2808, 996,  2670, 2178, 988,  1879, 544,  1689, 1979, 2776, 2753, 3161, 2315, 711,  3136, 1880,
      1320, 1216, 2233, 2373, 2832, 328,  2009, 119,  297,  1304, 2334, 1493, 481,  2319, 2954, 1298, 1612, 3230, 197,
      2697, 1238, 1959, 2943, 3309, 1806, 2471, 2595, 50,   601,  1399, 876,  853,  360,  153,  849,  216,  1148, 891,
      2734, 3197, 242,  845,  219,  1700, 1615, 824,  1662, 2914, 1152, 1776, 1854, 1986, 222,  1287, 434,  598,  3296,
      869,  62,   2282, 1138, 1243, 2333, 2124, 1736, 2259, 2514, 1491, 2891, 911,  187,  1622, 3249, 1489, 1666, 2072,
      980,  516,  727,  581,  672,  3161, 622,  2033, 2525, 3180, 1362, 1823, 439,  2921, 1021, 117,  1014, 936,  1656,
      2610, 2411, 3024, 1842, 2865, 1023, 2933, 39,   596,  3255, 3256, 1077, 2914, 2382, 911,  863,  999,  1594, 1668,
      377,  14,   1687, 165,  3143, 3,    61,   1395, 983,  3218, 2366, 3296, 1682, 66,   448,  1391, 1543, 1588, 1084,
      895,  1053, 2185, 3290, 2672, 2939, 1353, 2161, 2260, 465,  5,    2617, 2345, 2333, 1945, 2321, 2979, 25,   3006,
      2779, 2858, 1139, 1084, 2629, 1599, 1885, 762,  2461, 2315, 485,  2730, 3202, 3197, 1352, 1608, 1790, 1549, 3127,
      2479, 2728, 2485, 1426, 247,  168,  2178, 32,   3042},
     {2704, 2318, 284,  2909, 895,  894,  1184, 2656, 2487, 64,   2579, 793,  233,  2008, 2131, 870,  2995, 1663, 1642,
      1039, 391,  1338, 2950, 2309, 43,   1470, 721,  1628, 2232, 3002, 1030, 2042, 1486, 478,  65,   2912, 1956, 1585,
      1728, 892,  2034, 561,  2858, 2208, 825,  263,  2637, 2924, 2688, 499,  1673, 2923, 2504, 1316, 2017, 2953, 1514,
      970,  2212, 1587, 1573, 3200, 2000, 684,  639,  427,  416,  1367, 2070, 1112, 27,   70,   418,  514,  397,  231,
      2314, 1243, 2796, 1904, 2562, 3084, 2436, 354,  935,  3078, 244,  2468, 689,  1008, 3189, 779,  2934, 3073, 2366,
      793,  2781, 489,  1960, 1893, 1143, 2520, 2555, 1016, 497,  2069, 634,  2419, 238,  2694, 1287, 2828, 2763, 1307,
      1036, 623,  2808, 168,  3020, 1464, 910,  1745, 186,  957,  2181, 2059, 775,  1392, 2297, 1204, 1559, 260,  2750,
      3041, 1652, 739,  1193, 175,  3,    3239, 2318, 2206, 468,  1996, 583,  3202, 3165, 150,  892,  2860, 3126, 263,
      2629, 1626, 544,  629,  1631, 759,  380,  3210, 2238, 2615, 400,  2786, 2961, 749,  365,  203,  635,  1927, 1222,
      1932, 966,  382,  1709, 1539, 2503, 1518, 591,  1628, 1319, 1387, 2606, 138,  302,  2261, 2594, 2485, 1153, 387,
      3054, 1109, 2259, 284,  553,  2728, 3162, 562,  888,  949,  1562, 2423, 1209, 2869, 476,  136,  1924, 1257, 851,
      2307, 3136, 1027, 2541, 1675, 30,   417,  1181, 3017, 297,  2541, 1050, 1483, 930,  697,  1032, 2604, 3180, 636,
      1727, 50,   89,   1625, 1725, 1296, 2033, 1725, 3225, 2471, 1284, 1782, 1306, 814,  3250, 2326, 3298, 2964, 1715,
      3265, 2519, 2874, 1171, 781,  332,  3310, 1976, 94},
     {3294, 1948, 588,  2221, 3162, 2400, 431,  1807, 372,  3200, 1581, 703,  44,   2582, 1879, 2348, 2768, 795,  15,
      2746, 180,  151,  903,  642,  2457, 2400, 3081, 2107, 1152, 1334, 2677, 283,  752,  3018, 2868, 296,  1624, 3243,
      2398, 942,  2720, 1638, 713,  2466, 90,   2341, 1738, 2175, 413,  328,  1384, 746,  540,  2848, 894,  3096, 375,
      344,  3258, 1879, 2700, 226,  2868, 1451, 2078, 42,   602,  2276, 1623, 2651, 2175, 3325, 1806, 1542, 2566, 2410,
      3234, 2975, 96,   2612, 246,  630,  2329, 3217, 836,  2900, 2673, 2691, 1711, 1588, 1867, 2076, 1987, 1579, 2416,
      2424, 2904, 684,  432,  272,  1942, 2076, 2424, 2086, 1037, 653,  701,  881,  1367, 531,  3138, 542,  631,  2236,
      955,  2656, 2813, 2313, 1206, 422,  2424, 1168, 404,  3047, 859,  3220, 1072, 2056, 3003, 792,  3049, 1914, 1325,
      432,  2949, 1561, 3291, 491,  2557, 1749, 1631, 39,   1362, 2806, 497,  2845, 1580, 1053, 1486, 752,  2418, 797,
      949,  275,  76,   32,   2804, 286,  1483, 1828, 2888, 2795, 1246, 2875, 3146, 1490, 1756, 969,  36,   1503, 89,
      1474, 1596, 1364, 1122, 2301, 530,  1898, 2526, 969,  36,   1656, 2812, 469,  1198, 2835, 1032, 3314, 1771, 3091,
      1078, 3214, 1927, 1953, 1252, 1037, 470,  165,  900,  636,  2150, 2129, 270,  1885, 344,  528,  385,  334,  2293,
      3217, 2017, 2093, 1128, 2506, 3328, 2522, 2758, 3060, 48,   981,  989,  706,  2540, 2139, 1025, 1616, 2938, 2975,
      737,  1641, 1140, 2959, 2336, 1083, 1504, 1133, 2749, 1433, 382,  2115, 3157, 2406, 1542, 2066, 1431, 2932, 738,
      350,  470,  2680, 1627, 2115, 2160, 250,  956,  1412}},
    {{1595, 1348, 2434, 1980, 566,  2509, 2716, 734,  3043, 2490, 227,  3286, 2944, 833,  269,  2901, 148,  1731, 3155,
      2630, 1073, 1437, 3018, 733,  2221, 608,  2454, 586,  604,  3038, 1605, 622,  563,  1136, 2161, 472,  231,  541,
      2337, 2075, 2940, 1891, 122,  1683, 2357, 1841, 1030, 1219, 602,  642,  3198, 1642, 3297, 35,   557,  3166, 2624,
      1740, 2104, 1552, 2419, 2344, 2887, 1119, 1570, 3041, 1473, 3286, 152,  1536, 2862, 200,  3263, 1335, 2161, 1565,
      204,  1992, 2701, 1752, 1662, 295,  1475, 365,  27,   3065, 1977, 295,  1779, 889,  112,  1454, 3181, 641,  709,
      2703, 2162, 1673, 799,  1695, 282,  2540, 911,  2720, 360,  2426, 1789, 1562, 3108, 444,  3150, 1411, 1983, 288,
      1594, 1165, 2575, 659,  2150, 678,  1008, 1656, 2253, 746,  2824, 1325, 2967, 313,  275,  1576, 3302, 2333, 664,
      1978, 1081, 1542, 1125, 451,  366,  1180, 1283, 2066, 173,  907,  222,  2661, 1850, 3207, 336,  3166, 1136, 450,
      1842, 840,  1930, 3028, 1806, 1099, 1557, 2776, 546,  1062, 1019, 2846, 514,  1206, 3053, 942,  1539, 1828, 1631,
      972,  2862, 1876, 859,  1809, 273,  2665, 1454, 2596, 1881, 422,  2479, 2806, 1760, 810,  2303, 1247, 2659, 2955,
      2606, 766,  694,  1404, 3071, 2593, 2424, 3117, 2060, 2175, 1444, 1392, 3224, 615,  1852, 1970, 2039, 876,  158,
      221,  2221, 1370, 748,  2090, 2922, 2533, 1061, 389,  2067, 2017, 609,  1205, 1832, 339,  2309, 2419, 3009, 490,
      2617, 2094, 1521, 228,  1491, 478,  516,  3227, 234,  1681, 2056, 1494, 290,  835,  1436, 549,  1903, 904,  1039,
      1019, 531,  2618, 1255, 785,  2779, 2885, 2893, 2524},
     {2810, 1820, 3275, 3022, 1194, 926,  3191, 3147, 3268, 2418, 2493, 1433, 1014, 522,  2746, 860,  1636, 786,  2910,
      454,  2900, 1243, 3189, 18,   2662, 1591, 2696, 871,  1650, 2396, 832,  3270, 2686, 103,  995,  2657, 1121, 665,
      1218, 2061, 600,  1550, 731,  1162, 2400, 1438, 351,  299,  3018, 2590, 1010, 1930, 1954, 486,  1509, 1713, 111,
      2625, 1139, 41,   488,  2353, 1650, 978,  2117, 1995, 2519, 1867, 1398, 2707, 199,  3096, 653,  141,  872,  2259,
      1079, 1962, 1302, 468,  3120, 2787, 533,  731,  1235, 1882, 1244, 2253, 2271, 703,  1067, 1778, 1130, 1668, 2509,
      3004, 2675, 97,   1174, 2201, 1631, 1032, 805,  766,  3304, 659,  2852, 2267, 2493, 3013, 2555, 605,  2240, 1181,
      1909, 3023, 1822, 2338, 2975, 258,  409,  2438, 709,  2546, 1804, 676,  1849, 691,  1365, 2023, 701,  1827, 2839,
      3062, 1055, 921,  1900, 1812, 2071, 1044, 2026, 2191, 959,  1792, 2897, 1005, 2055, 375,  2772, 1748, 326,  1825,
      129,  1253, 3281, 1058, 1420, 1575, 1901, 917,  604,  2702, 2335, 3250, 1058, 1261, 2497, 2725, 1803, 549,  2336,
      2528, 1004, 1853, 3295, 1474, 1753, 2603, 123,  2268, 2325, 2857, 472,  350,  3307, 1307, 2495, 1363, 2814, 180,
      1820, 1489, 2414, 1520, 2338, 773,  1069, 2325, 1133, 2334, 2220, 1484, 1928, 2988, 719,  3264, 2866, 2674, 268,
      3194, 558,  1085, 1013, 541,  3227, 290,  1659, 1414, 41,   105,  2255, 115,  2604, 3000, 560,  2982, 2068, 992,
      1116, 381,  2439, 1723, 1704, 2255, 1140, 2201, 1588, 2911, 1833, 847,  1851, 24,   850,  708,  1845, 1269, 2945,
      49,   1487, 2886, 430,  537,  2736, 2161, 1617, 1812},
     {1389, 3238, 2885, 64,   418,  942,  1221, 1404, 330,  2839, 1581, 1845, 1567, 1970, 153,  1579, 1383, 10,   1500,
      1539, 128,  2479, 2070, 2669, 542,  3126, 1840, 2522, 553,  533,  400,  1493, 1545, 1318, 704,  793,  1735, 1124,
      743,  1122, 1675, 3272, 2651, 2255, 742,  1493, 482,  124,  2435, 1489, 205,  2792, 1291, 1918, 178,  2941, 1280,
      1883, 2612, 1901, 901,  1804, 1364, 1377, 2363, 88,   1966, 632,  1975, 14,   504,  455,  2646, 1016, 1096, 2943,
      2589, 3034, 2384, 2308, 933,  52,   621,  1550, 729,  661,  295,  83,   2105, 3308, 2211, 3183, 684,  3255, 2175,
      1158, 1845, 78,   2054, 3112, 2887, 1337, 2152, 2874, 3116, 783,  1061, 1241, 2835, 2704, 1060, 650,  2769, 1848,
      2441, 1900, 1516, 1297, 3057, 2945, 1995, 1619, 2711, 2317, 2346, 280,  2159, 3303, 1022, 1285, 1067, 442,  2901,
      89,   2911, 1285, 2875, 42,   646,  1855, 2440, 1171, 553,  1332, 164,  200,  2526, 406,  720,  499,  2899, 991,
      1552, 279,  518,  2298, 1726, 453,  2849, 136,  914,  1049, 110,  898,  1879, 2827, 1979, 2913, 3109, 23,   3179,
      2025, 2406, 1721, 702,  2365, 3194, 2480, 3249, 1606, 369,  1929, 662,  773,  1913, 1411, 1547, 92,   246,  877,
      790,  1100, 2168, 423,  2612, 2774, 1106, 1485, 1046, 895,  1126, 2531, 2403, 2615, 1377, 1006, 354,  1531, 2373,
      80,   2154, 1282, 341,  79,   227,  1479, 2979, 2217, 2836, 3036, 530,  328,  1116, 272,  1524, 763,  2512, 1076,
      1156, 2205, 1506, 485,  2621, 2529, 2239, 566,  2442, 2786, 229,  195,  1917, 1384, 2674, 2533, 2069, 2182, 3105,
      197,  3242, 3061, 315,  2422, 2725, 1633, 1975, 125}},
    {{810,  3174, 2638, 2440, 2357, 522,  919,  2319, 1192, 2899, 3205, 3239, 644,  2773, 2840, 1838, 749,  2563, 818,
      1246, 2264, 784,  191,  576,  2512, 240,  858,  221,  2438, 1536, 1753, 205,  1859, 1027, 1054, 2369, 670,  66,
      769,  336,  1469, 3243, 1294, 1896, 1859, 2171, 109,  2489, 3315, 128,  789,  3070, 194,  3283, 403,  2294, 475,
      2713, 2420, 1504, 1173, 257,  3259, 599,  1813, 772,  3101, 2417, 1790, 45,   530,  117,  191,  419,  3207, 1289,
      1170, 2616, 741,  447,  442,  844,  637,  92,   1100, 1889, 3279, 3264, 2658, 1107, 3165, 387,  2673, 1668, 2292,
      3004, 2773, 1074, 1044, 3244, 2874, 483,  49,   377,  1427, 1007, 541,  1227, 175,  357,  319,  628,  2010, 599,
      2878, 921,  229,  946,  2413, 723,  2431, 2090, 318,  3179, 2073, 201,  2171, 1781, 319,  1418, 2429, 685,  829,
      10,   969,  85,   303,  1346, 3313, 2015, 1902, 2557, 3051, 540,  439,  2592, 1940, 790,  1659, 3130, 296,  2311,
      1007, 2885, 2712, 2197, 248,  2256, 105,  1438, 671,  182,  856,  162,  2954, 2271, 1893, 2575, 2473, 2701, 267,
      2719, 3287, 2672, 2971, 3008, 1012, 240,  1781, 2792, 542,  3236, 2514, 369,  1349, 1345, 3213, 2319, 161,  362,
      1377, 145,  1591, 2886, 2346, 2371, 1628, 247,  1462, 759,  3130, 1640, 1305, 2256, 372,  643,  1397, 532,  2009,
      3126, 1033, 2819, 806,  1165, 2935, 162,  2312, 2194, 1175, 3276, 251,  2125, 2650, 840,  1713, 3106, 1313, 2789,
      2803, 1587, 1977, 450,  643,  238,  1958, 1189, 673,  105,  995,  2259, 123,  1182, 653,  430,  906,  1493, 1275,
      1666, 791,  1994, 2573, 35,   727,  1839, 1381, 257},
     {808,  2243, 2180, 5,    277,  65,   1953, 1605, 3270, 3013, 783,  17,   2600, 728,  845,  1828, 392,  2838, 374,
      1999, 3092, 814,  1923, 3174, 1018, 1218, 2555, 62,   1122, 83,   1368, 561,  501,  1744, 2185, 1810, 2368, 2430,
      229,  870,  1391, 3285, 1093, 110,  180,  1961, 2765, 640,  536,  1379, 611,  3060, 935,  1142, 701,  2758, 147,
      2233, 269,  108,  2637, 513,  2305, 2039, 2765, 2133, 2194, 929,  156,  1738, 955,  1042, 2870, 994,  3213, 1752,
      2554, 2906, 749,  620,  2072, 2949, 3286, 3061, 253,  1992, 1786, 956,  2486, 20,   3155, 2298, 42,   595,  2643,
      1429, 1567, 1264, 1208, 880,  1935, 562,  1688, 20,   1970, 1953, 2249, 1181, 1132, 32,   1364, 1400, 1540, 2634,
      2909, 1797, 2039, 2027, 1662, 2562, 2170, 1416, 2387, 1272, 194,  1183, 2038, 2840, 2583, 174,  2807, 417,  979,
      1314, 2869, 219,  1876, 895,  2275, 719,  2433, 2802, 884,  1635, 2934, 2978, 142,  1953, 2620, 2020, 1076, 829,
      224,  803,  2914, 3035, 442,  954,  2000, 1093, 3104, 3064, 1049, 2286, 1233, 356,  2354, 1501, 474,  1993, 1895,
      2505, 667,  2879, 1833, 1167, 973,  2082, 1934, 3236, 663,  536,  863,  1071, 1292, 193,  951,  450,  2830, 1453,
      186,  392,  774,  3237, 782,  54,   64,   1846, 678,  250,  1432, 1183, 520,  3178, 892,  2497, 824,  1024, 1078,
      783,  1897, 260,  1796, 979,  2382, 1581, 2388, 1966, 1804, 2233, 1933, 463,  1358, 1918, 1400, 1576, 1069, 582,
      3023, 2020, 1356, 846,  608,  3134, 1209, 2797, 1386, 2926, 1619, 2610, 1969, 555,  540,  3325, 2972, 2291, 3187,
      1257, 2598, 597,  2919, 1074, 2983, 3188, 868,  1246},
     {1692, 1998, 1422, 3267, 924,  2791, 314,  3014, 2402, 1818, 1550, 2452, 3061, 2,    25,   2004, 2384, 905,  1222,
      1242, 2219, 5,    1825, 102,  692,  2977, 1386, 2237, 181,  1545, 617,  867,  3014, 2109, 2293, 496,  2721, 2646,
      1093, 700,  2660, 1268, 629,  2956, 3307, 2767, 490,  1021, 3024, 1680, 2608, 2573, 1634, 2230, 112,  2579, 3221,
      683,  1297, 1363, 661,  787,  2149, 2947, 1065, 2785, 2453, 2353, 1360, 2112, 3139, 2949, 1401, 886,  1833, 1029,
      1545, 1143, 14,   3240, 2166, 1098, 2751, 2956, 2786, 2474, 1884, 1680, 2431, 3254, 1657, 2310, 2170, 201,  563,
      621,  1406, 1741, 425,  1051, 276,  2435, 1798, 1999, 2026, 1161, 1111, 425,  1004, 511,  983,  2871, 381,  905,
      2789, 3099, 2625, 978,  865,  1914, 996,  2790, 1576, 260,  2962, 211,  82,   2339, 2970, 3311, 798,  1570, 1812,
      1677, 3092, 2872, 515,  2103, 344,  1554, 2495, 2696, 718,  2796, 313,  354,  354,  2840, 1336, 2673, 2665, 2108,
      2986, 2872, 2283, 3204, 3235, 1617, 1971, 2650, 1523, 3156, 2058, 2062, 1299, 146,  2491, 1652, 90,   2633, 1902,
      1905, 1465, 2551, 407,  3293, 2655, 2179, 22,   0,    2747, 3231, 515,  341,  197,  3034, 3097, 2558, 2010, 3143,
      1168, 2401, 7,    3086, 1432, 2788, 1336, 987,  2480, 2832, 1391, 3152, 646,  3039, 1332, 2707, 1250, 488,  1045,
      882,  2194, 2660, 802,  1043, 801,  974,  1937, 330,  951,  333,  1323, 547,  2505, 3124, 1475, 2061, 2098, 3285,
      1132, 1320, 3065, 2113, 3104, 2160, 3089, 3240, 894,  2396, 474,  1791, 1620, 1036, 2863, 508,  134,  2938, 3243,
      1030, 1010, 779,  529,  1037, 367,  668,  1602, 563}}};

  const int KYBER_K = 3;

  // Allocate device memory
  uint64_t* d_rho;
  Zq* d_A;
  cudaMalloc(&d_rho, 4 * sizeof(uint64_t));
  cudaMalloc(&d_A, KYBER_K * KYBER_K * 256 * sizeof(Zq));

  // Convert hex string to bytes
  std::vector<uint8_t> rho_bytes(32);
  for (int i = 0; i < 32; i++) {
    rho_bytes[i] = std::stoi(rho.substr(i * 2, 2), nullptr, 16);
  }

  // Copy rho to device as uint64_t array
  uint64_t h_rho[4];
  memcpy(h_rho, rho_bytes.data(), 32);
  cudaMemcpy(d_rho, h_rho, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Run kernel
  sampleA_test_kernel<KYBER_K><<<1, 128>>>(d_rho, d_A);

  // Copy results back
  uint16_t result[KYBER_K * KYBER_K][256];
  cudaMemcpy(result, d_A, KYBER_K * KYBER_K * 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

  printf("start verifying\n");

  // Verify results
  for (int i = 0; i < KYBER_K; i++) {
    for (int j = 0; j < KYBER_K; j++) {
      for (int k = 0; k < 256; k++) {
        ASSERT_EQ(A[i][j][k], result[i * KYBER_K + j][k]) << "Mismatch at A[" << i << "][" << j << "][" << k << "]";
      }
    }
  }

  // Cleanup
  cudaFree(d_rho);
  cudaFree(d_A);
}

TEST_F(KyberTest, SamplePolyCBD_2_768)
{
  const std::string sigma = "b7ef1194d28c94cade1217b9ff402ed1165707fdbff655f04bf2f5981d830be1";

  const int KYBER_K = 3;

  uint16_t s[KYBER_K][256] = {
    {0,    1,    1,    0,    0,    0,    0,    0,    2,    1,    3328, 3328, 0,    0,    0,    0,    0,    0,    3328,
     0,    1,    3328, 0,    3328, 3328, 3328, 2,    3327, 1,    3328, 3328, 0,    1,    3328, 0,    2,    2,    1,
     3328, 3328, 0,    3328, 1,    1,    1,    2,    3328, 0,    0,    1,    1,    3327, 0,    3328, 3327, 3328, 0,
     3328, 2,    0,    3328, 0,    1,    1,    1,    1,    0,    3328, 2,    0,    1,    1,    3328, 1,    3327, 0,
     0,    1,    0,    3328, 3328, 3327, 0,    1,    0,    0,    0,    3328, 0,    0,    0,    3328, 0,    3328, 1,
     0,    3328, 3328, 3327, 1,    0,    1,    0,    0,    3328, 0,    1,    1,    0,    0,    1,    3328, 3328, 3328,
     1,    0,    3328, 3328, 3328, 3328, 1,    0,    3328, 2,    1,    0,    3328, 1,    1,    0,    1,    3327, 0,
     0,    0,    3328, 1,    3328, 0,    0,    3328, 0,    0,    3328, 3328, 2,    2,    3327, 0,    3328, 3328, 3327,
     3328, 3328, 0,    0,    1,    0,    0,    0,    1,    0,    1,    1,    0,    0,    0,    3328, 0,    2,    1,
     0,    3328, 3328, 1,    0,    0,    0,    2,    3328, 3328, 1,    3328, 1,    0,    1,    3328, 1,    0,    1,
     0,    0,    2,    2,    1,    3328, 3328, 3327, 0,    0,    3328, 3328, 1,    2,    0,    0,    0,    3328, 3328,
     1,    1,    0,    0,    2,    3328, 0,    1,    1,    3328, 3327, 1,    1,    3328, 0,    3328, 3327, 3328, 0,
     0,    3328, 3328, 1,    3328, 0,    1,    3327, 0,    1,    0,    2,    0,    3328, 0,    0,    1,    1,    3328,
     3328, 3328, 3328, 2,    0,    0,    1,    3328, 3328},
    {0,    3327, 3327, 0,    3328, 0,    1,    0,    1,    0,    0,    1,    0,    0,    0,    0,    0,    2,    3328,
     1,    3328, 0,    0,    1,    0,    0,    0,    3327, 1,    3328, 3328, 2,    3328, 3328, 1,    3327, 2,    3328,
     1,    3327, 3328, 1,    1,    3328, 1,    1,    0,    0,    0,    3328, 0,    1,    1,    3328, 3328, 3328, 0,
     0,    1,    0,    0,    0,    0,    3328, 3328, 0,    0,    1,    2,    1,    0,    1,    2,    0,    0,    3328,
     0,    3328, 3328, 1,    0,    0,    3328, 0,    0,    0,    1,    1,    1,    0,    3328, 3328, 0,    2,    1,
     3328, 0,    1,    3328, 3328, 2,    3328, 1,    2,    3328, 3328, 0,    3328, 3328, 0,    0,    1,    0,    0,
     1,    3328, 0,    1,    2,    1,    3328, 3328, 3328, 0,    3328, 1,    1,    1,    3327, 1,    1,    0,    3327,
     2,    0,    1,    1,    3328, 1,    1,    1,    1,    3328, 2,    0,    1,    1,    3327, 1,    0,    3328, 2,
     0,    3328, 3328, 0,    1,    3328, 1,    2,    3328, 0,    0,    0,    0,    0,    3327, 1,    0,    0,    1,
     1,    2,    3328, 3328, 3328, 0,    0,    1,    0,    3328, 1,    3328, 3328, 3328, 2,    3328, 0,    0,    0,
     3328, 3328, 0,    3327, 0,    3328, 1,    1,    3328, 0,    0,    3328, 1,    3327, 1,    1,    0,    3328, 3328,
     0,    3328, 1,    3328, 1,    1,    3328, 0,    1,    1,    0,    0,    0,    0,    0,    3328, 1,    0,    1,
     0,    0,    3328, 3328, 0,    3327, 3328, 0,    0,    3328, 0,    0,    3328, 0,    0,    3328, 3328, 1,    1,
     3328, 3328, 3327, 1,    3328, 0,    1,    2,    3327},
    {1,    3328, 0,    1,    3328, 0,    3328, 3328, 0,    2,    0,    0,    2,    0,    3328, 0,    1,    3328, 0,
     1,    1,    3328, 3328, 0,    0,    3328, 1,    3328, 2,    0,    0,    1,    3328, 0,    2,    1,    3328, 1,
     3328, 3328, 3328, 1,    2,    3328, 1,    2,    1,    0,    3328, 0,    0,    3327, 2,    0,    0,    1,    1,
     3328, 3328, 0,    1,    0,    1,    3328, 1,    3328, 3328, 3328, 1,    1,    0,    3328, 1,    0,    3327, 2,
     1,    1,    1,    3328, 0,    3328, 1,    3328, 3327, 0,    0,    0,    3328, 0,    2,    3328, 1,    3328, 3328,
     0,    1,    1,    0,    0,    3327, 3327, 0,    1,    0,    1,    1,    1,    3328, 3328, 1,    0,    0,    0,
     0,    0,    0,    1,    1,    1,    2,    3328, 3328, 1,    1,    1,    0,    0,    1,    1,    0,    1,    1,
     1,    1,    0,    3328, 0,    0,    0,    0,    0,    1,    0,    3328, 1,    2,    0,    2,    3328, 1,    2,
     3328, 0,    0,    1,    0,    0,    0,    1,    0,    1,    3328, 0,    0,    1,    3328, 1,    3328, 0,    0,
     0,    1,    3328, 1,    3327, 2,    1,    2,    1,    3327, 0,    1,    0,    1,    2,    1,    3327, 0,    1,
     0,    0,    0,    0,    0,    2,    0,    2,    1,    1,    2,    0,    1,    3328, 0,    3328, 3328, 3328, 1,
     1,    1,    0,    0,    1,    0,    1,    3328, 1,    0,    1,    0,    2,    3327, 1,    0,    1,    1,    3328,
     2,    3327, 0,    1,    1,    3328, 1,    1,    3328, 0,    3328, 0,    2,    3328, 1,    1,    3328, 0,    0,
     0,    0,    0,    3327, 0,    3327, 0,    0,    0}};

  uint16_t e[KYBER_K][256] = {
    {0,    3328, 1,    0,    1,    1,    3328, 0,    3328, 3328, 3328, 1,    3328, 1,    0,    0,    1,    3328, 0,
     3328, 0,    0,    1,    0,    0,    1,    1,    2,    0,    3328, 0,    0,    1,    0,    0,    0,    1,    1,
     0,    3328, 1,    0,    0,    0,    1,    1,    3328, 0,    0,    2,    0,    2,    3328, 0,    1,    1,    0,
     1,    3328, 0,    2,    3328, 1,    0,    3328, 2,    0,    3328, 0,    3327, 3328, 3328, 0,    0,    2,    2,
     1,    3328, 0,    1,    3328, 0,    0,    1,    1,    3327, 2,    0,    0,    0,    3328, 0,    0,    0,    1,
     0,    2,    1,    1,    0,    3328, 3327, 0,    0,    1,    0,    0,    3328, 0,    3328, 0,    2,    0,    0,
     3328, 2,    1,    3328, 3328, 3328, 2,    3327, 0,    0,    0,    3328, 2,    3328, 1,    3328, 3327, 2,    1,
     0,    3328, 1,    3328, 3328, 0,    2,    3328, 2,    3327, 0,    1,    3328, 1,    3328, 0,    3328, 0,    0,
     1,    3328, 0,    2,    1,    3328, 3328, 0,    3328, 3328, 2,    1,    3327, 3328, 3328, 1,    1,    1,    0,
     0,    1,    0,    1,    3328, 0,    2,    2,    3328, 1,    2,    1,    3328, 0,    0,    3328, 1,    0,    0,
     1,    1,    1,    3328, 3328, 0,    0,    0,    0,    0,    1,    1,    1,    1,    0,    3328, 0,    0,    2,
     0,    1,    2,    0,    3328, 0,    0,    0,    3328, 0,    0,    0,    3328, 3327, 3328, 1,    0,    0,    0,
     1,    1,    3328, 0,    0,    3328, 0,    1,    1,    3328, 2,    2,    0,    0,    3328, 3328, 0,    1,    0,
     3328, 3328, 1,    0,    3328, 3328, 3328, 1,    3328},
    {3328, 0,    0,    0,    3328, 0,    0,    3327, 0,    1,    0,    0,    1,    0,    0,    0,    3328, 1,    3328,
     0,    3327, 3327, 1,    0,    3328, 3328, 1,    1,    3328, 0,    1,    3328, 1,    0,    3328, 3328, 0,    1,
     3327, 3328, 3328, 3328, 2,    0,    1,    0,    0,    0,    1,    1,    0,    1,    1,    1,    3328, 0,    0,
     3328, 0,    0,    0,    3328, 1,    2,    0,    3327, 1,    3328, 3328, 0,    0,    1,    3328, 0,    3328, 2,
     1,    3328, 0,    0,    3328, 2,    0,    0,    0,    3328, 0,    3328, 0,    3328, 0,    1,    3328, 3328, 1,
     1,    1,    0,    3328, 1,    3328, 1,    3328, 3327, 1,    3328, 0,    1,    2,    0,    3327, 3328, 1,    0,
     1,    1,    3328, 3328, 3328, 1,    3328, 2,    0,    0,    1,    0,    0,    1,    3328, 1,    1,    3328, 1,
     1,    1,    3328, 0,    3328, 1,    0,    1,    1,    3327, 3327, 0,    0,    0,    1,    1,    3328, 0,    0,
     1,    1,    1,    0,    3328, 1,    0,    1,    0,    0,    3328, 0,    0,    0,    0,    3328, 0,    1,    2,
     0,    3328, 3328, 0,    1,    0,    0,    3328, 1,    0,    1,    0,    1,    0,    0,    3328, 1,    1,    0,
     0,    1,    0,    1,    0,    1,    0,    0,    3328, 3328, 0,    0,    0,    3328, 3328, 3328, 1,    1,    0,
     2,    1,    0,    0,    3328, 3328, 3328, 3328, 0,    3328, 3327, 3327, 0,    2,    1,    0,    0,    0,    0,
     2,    0,    3328, 1,    0,    1,    0,    3328, 1,    0,    2,    3328, 0,    0,    3328, 3328, 3328, 0,    3328,
     0,    3328, 3328, 3328, 3327, 0,    2,    1,    0},
    {1,    0,    1,    0,    1,    2,    3328, 0,    3328, 3328, 1,    3328, 0,    3328, 0,    0,    0,    0,    3328,
     0,    0,    1,    0,    1,    0,    3328, 3328, 1,    3328, 0,    3327, 3328, 0,    0,    3328, 2,    3328, 3328,
     0,    3327, 0,    0,    0,    1,    2,    3328, 0,    0,    0,    0,    1,    3328, 0,    0,    0,    3328, 3327,
     3328, 1,    3327, 0,    0,    0,    0,    0,    3328, 0,    2,    0,    0,    2,    1,    3328, 1,    2,    0,
     2,    1,    1,    1,    3328, 3328, 3328, 3328, 0,    0,    0,    0,    1,    2,    1,    1,    3328, 3328, 3328,
     1,    0,    3328, 0,    3328, 0,    3327, 1,    1,    0,    0,    3328, 3328, 3328, 1,    3328, 1,    0,    1,
     0,    3328, 1,    3328, 1,    1,    3328, 3327, 3328, 0,    1,    1,    0,    3328, 0,    3328, 3327, 3327, 2,
     0,    3328, 0,    3327, 0,    1,    3328, 3328, 0,    1,    0,    0,    2,    3327, 1,    0,    3327, 0,    1,
     0,    3327, 0,    3327, 3328, 0,    0,    0,    1,    0,    0,    0,    3327, 0,    0,    3328, 0,    1,    0,
     3328, 3328, 3328, 1,    0,    3328, 0,    3328, 0,    0,    1,    0,    3328, 3328, 0,    3328, 3327, 3328, 3328,
     3328, 3328, 1,    0,    3328, 3328, 1,    0,    0,    3328, 1,    2,    3328, 0,    0,    3328, 2,    3328, 2,
     1,    3328, 1,    3328, 1,    1,    1,    0,    3327, 1,    0,    3328, 0,    3328, 0,    0,    0,    1,    0,
     3327, 3328, 0,    0,    3327, 3328, 0,    0,    1,    0,    3328, 0,    0,    0,    3328, 0,    0,    1,    0,
     3328, 1,    0,    3328, 3327, 3328, 1,    0,    0}};

  // Convert sigma hex string to bytes
  std::array<uint8_t, 32> sigma_bytes;
  for (int i = 0; i < 32; i++) {
    sigma_bytes[i] = std::stoi(sigma.substr(i * 2, 2), nullptr, 16);
  }

  // Copy sigma to device as uint64_t array
  uint64_t h_sigma[4];
  memcpy(h_sigma, sigma_bytes.data(), 32);
  uint64_t* d_sigma;
  cudaMalloc((void**)&d_sigma, 4 * sizeof(uint64_t));
  cudaMemcpy(d_sigma, h_sigma, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Allocate device memory for s and e as one contiguous array
  Zq* d_se;
  cudaMalloc((void**)&d_se, KYBER_K * 2 * 256 * sizeof(Zq)); // Allocate space for both s and e
  Zq* d_s = d_se;                                            // First half is s
  Zq* d_e = d_se + (KYBER_K * 256);                          // Second half is e

  // Run kernel to sample s and e
  samplePolyCBD_test_kernel<KYBER_K, 2><<<1, 128>>>(d_sigma, d_se);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Copy results back
  uint16_t result_s[KYBER_K][256];
  uint16_t result_e[KYBER_K][256];
  cudaMemcpy(result_s, d_s, KYBER_K * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);
  cudaMemcpy(result_e, d_e, KYBER_K * 256 * sizeof(Zq), cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < KYBER_K; i++) {
    for (int j = 0; j < 256; j++) {
      ASSERT_EQ(s[i][j], result_s[i][j]) << "Mismatch at s[" << i << "][" << j << "]";
      ASSERT_EQ(e[i][j], result_e[i][j]) << "Mismatch at e[" << i << "][" << j << "]";
    }
  }

  // Cleanup
  cudaFree(d_sigma);
  cudaFree(d_se); // Only need to free the base pointer
}

TEST_F(KyberTest, PkeKeygen768Batch)
{
  const uint64_t batch = 2048 * 4;
  // const uint64_t batch = 2048;
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const uint8_t KYBER_K = 3;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 384 * KYBER_K;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate memory for batch
  cudaMalloc((void**)&d_d, 32 * batch);
  cudaMalloc((void**)&d_ek, ek_size * batch);
  cudaMalloc((void**)&d_dk, dk_size * batch);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq) * batch);

  // Generate random input data
  uint8_t* h_d = new uint8_t[32 * batch];
  for (uint64_t i = 0; i < 32 * batch; i++) {
    h_d[i] = rand() % 256;
  }
  cudaMemcpy(d_d, h_d, 32 * batch, cudaMemcpyHostToDevice);

  // Launch kernel for batch
  pke_keygen<KYBER_K, 2><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  // pke_keygen<KYBER_K, 3><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Cleanup
  delete[] h_d;
  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, PkeKeygen1024Batch)
{
  const uint64_t batch = 2048 * 4;
  // const uint64_t batch = 2048;
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const uint8_t KYBER_K = 4;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 384 * KYBER_K;
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate memory for batch
  cudaMalloc((void**)&d_d, 32 * batch);
  cudaMalloc((void**)&d_ek, ek_size * batch);
  cudaMalloc((void**)&d_dk, dk_size * batch);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq) * batch);

  // Generate random input data
  uint8_t* h_d = new uint8_t[32 * batch];
  for (uint64_t i = 0; i < 32 * batch; i++) {
    h_d[i] = rand() % 256;
  }
  cudaMemcpy(d_d, h_d, 32 * batch, cudaMemcpyHostToDevice);

  // Launch kernel for batch
  pke_keygen<KYBER_K, 2><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  cudaDeviceSynchronize();

  // Cleanup
  delete[] h_d;
  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, ML_KEM_Internal_Keygen512)
{
  const std::string d = "3c53596d802b9371d9893e3822a35af22d3b872a2561f10b7072aced348c0d9d";
  const std::string z = "920c82b637ca89bb20c3dce155c7ffbedf0afd5972331cc72c30f315f6f5b04c";
  const std::string ek =
    "eb7b84f8b29729015487434804a5b700d7239e222d8b14b3352421c5e4283e06ab1ca82375b00a1ba3c6d2ec81bcf60a7f55bf6d2796d1947f"
    "c79724195abc89f91e4b923c4bb268eb6caa56d42d061ac323697092e57f71333bc54bc46a97ace8510e320972de819eb357674545b07ea504"
    "a6f2799898201485a2039574272cce0dcb4167fc0830900ad4b8a6e46932c4791ad99c709a8046985c0ba43bc4be722be8e04ea9f13bdb49a6"
    "25e7cb226738b251cabf168136e7837d4cbc9d884a5b6b27eb1774bae78617b22f443423f2c1b369f402a8253d2dfa58821290a8b89849b243"
    "66c4be8626790c86b485067649d5b91ae9127fa7bd353963d85c02d7d53a1cf919363068c6c6ca88fcc115bcc6e3568497668d8a771bf97322"
    "e6e523988c4e25aacce93986c72a0481cca855a07ebec44f4ce08cbcb87d92e855fa541bf95975f3ebc7531c2e5e751e02f2c587f959546a27"
    "7fe2a7d778b75251cc6d47308d936a543683a49624e28320e693bca7bab37b5a9cc16367fcb1296f8063cbe92520a2c58f20b698079c4acc04"
    "4a2c3170768f0214b2e01726b5d18b2834bec28856084ac3a9c836b21196a952cb755759be50753a2902a8a769f41a4889b3af56bb63a8c88b"
    "1694aa7304b95ce0c691f6411f9905c9738e0a170a9a7babc2027f0493168952bae24c2a98b84af7b009c8091a8873bde88066f4dabbe65346"
    "78ba2d8818ce25fa9488b4cb49b3c006e47e773485f5b5107b589b89a686512b597d8816aaf4971e74914625585240a0cc2a2e85329efc38b5"
    "53441c39155eb1a9293f660b2fa5467f4362e8b6460a07217ea84638e016c9d95835aa8f8784bf26a1cb56a863f105807ccc6457b85c191320"
    "16280fd49a105ec9ca91877381801fd557c107ba78cc18093a1031860512e4a6cbdc37394f4319ad90cc6c548fea49cebad0797eda3aaf8743"
    "5b7b20b6d8b16d4bbee37bcfa4e1af1c2760a3fa1e45983b79e8654f880ae5a8aa6a414e42c9b9e41c82895214f06a1ab5729d1b5ab546783c"
    "46bb87c7fbabdf9c7672c8a5c2a1b43758807e882c61fc3c135b9cc7a044deee0ed0375fc79324512e51fed475f5ff6fbbe692008884ea37ca"
    "ae2d";
  const std::string dk =
    "c06a8a83f66c3aaca2b8a3739e663e362c54ac9c6636131c77082438518d87d32c45829f02120bf0b0b4db4c85c492a3e5eab0d2c0828b115b"
    "30313629a4701c2b4e1b9cb872299fdae84ca6d190b99c43ea182f80b8950e12aee3b3022e908011c10d2fd2751d678f0dcc83ee34c45e344a"
    "cc39b374340b7951bffc00111e9123285cb906584890827211c6ab38872c523b698cb1c0fc8735c43263ca3b2142168df50ba0dc1072556087"
    "9a909554119cf40616411667e3e59c0b8c91ef93cd6f14553a70c8ecd16073f927c468a0cc39a4120b901fbc6fbae2a372a9baa1266adf35ab"
    "0057611de73537539333d0062af18fca68ba6c4a08d7b0c99b16b3c6111a29e33906aca499b6aa357b904dc37beda8241877aa40118f8f466f"
    "af90b7999c22965b697a58a1022212f10a2404289693185b40578c88e19e5e30ac11f00c63a5ae5320532350bac5710b0ab7763476c18bfb9f"
    "0aac7558a1bcc735237226b53486c07da88478903f3d36c92a6ac74b69cc3f3ac353e72b2e6441f62728f6d2ce4fd6568efb5d5934b4e05747"
    "6366bd11e321f610bfc29ba6d3b693b4ca4b54865d3f4912ddf9adb781723bf581e404312c7427289bb0f693710d8c55b4485daed5be9fd3ba"
    "348817840a4754a2357b5599da208cc301291ffcb3d7544d7b50578de45342b076d9c83165e5a5eee114c29607c0495414119fef5b8d8edab9"
    "f6b292117a21b8f6bfeb391b6d013e773b6076f1751e018faa832b1022728c91ab36a5ac98534751d93fd318403839c708979f6f7c18f3e24e"
    "fd505025eb9b0aa649a448904c62b82ff1c617e9b6823a2e900872c178bbe949bbe9a41c02d138094298e5fb11f1b5c81cf825a2dc255f549b"
    "2bbc97dfd92105bb42b7ca11c212a5df1c7e06654a5b580245805bc5a0b2a7e214593441fa7957120c2c51ba1ec39704846c5be877bd37843d"
    "47543c26a1c1b2b36923c2c4ebf5c935e3337dd88ed41a45b46246e5834b0b09b3030406275b887a69c1a200205ed89829307cc011b965f3c6"
    "7475c305787ea18a66f2b506b9b93c9ecb8402e52f93c10d67e2b6eb7b84f8b29729015487434804a5b700d7239e222d8b14b3352421c5e428"
    "3e06ab1ca82375b00a1ba3c6d2ec81bcf60a7f55bf6d2796d1947fc79724195abc89f91e4b923c4bb268eb6caa56d42d061ac323697092e57f"
    "71333bc54bc46a97ace8510e320972de819eb357674545b07ea504a6f2799898201485a2039574272cce0dcb4167fc0830900ad4b8a6e46932"
    "c4791ad99c709a8046985c0ba43bc4be722be8e04ea9f13bdb49a625e7cb226738b251cabf168136e7837d4cbc9d884a5b6b27eb1774bae786"
    "17b22f443423f2c1b369f402a8253d2dfa58821290a8b89849b24366c4be8626790c86b485067649d5b91ae9127fa7bd353963d85c02d7d53a"
    "1cf919363068c6c6ca88fcc115bcc6e3568497668d8a771bf97322e6e523988c4e25aacce93986c72a0481cca855a07ebec44f4ce08cbcb87d"
    "92e855fa541bf95975f3ebc7531c2e5e751e02f2c587f959546a277fe2a7d778b75251cc6d47308d936a543683a49624e28320e693bca7bab3"
    "7b5a9cc16367fcb1296f8063cbe92520a2c58f20b698079c4acc044a2c3170768f0214b2e01726b5d18b2834bec28856084ac3a9c836b21196"
    "a952cb755759be50753a2902a8a769f41a4889b3af56bb63a8c88b1694aa7304b95ce0c691f6411f9905c9738e0a170a9a7babc2027f049316"
    "8952bae24c2a98b84af7b009c8091a8873bde88066f4dabbe6534678ba2d8818ce25fa9488b4cb49b3c006e47e773485f5b5107b589b89a686"
    "512b597d8816aaf4971e74914625585240a0cc2a2e85329efc38b553441c39155eb1a9293f660b2fa5467f4362e8b6460a07217ea84638e016"
    "c9d95835aa8f8784bf26a1cb56a863f105807ccc6457b85c19132016280fd49a105ec9ca91877381801fd557c107ba78cc18093a1031860512"
    "e4a6cbdc37394f4319ad90cc6c548fea49cebad0797eda3aaf87435b7b20b6d8b16d4bbee37bcfa4e1af1c2760a3fa1e45983b79e8654f880a"
    "e5a8aa6a414e42c9b9e41c82895214f06a1ab5729d1b5ab546783c46bb87c7fbabdf9c7672c8a5c2a1b43758807e882c61fc3c135b9cc7a044"
    "deee0ed0375fc79324512e51fed475f5ff6fbbe692008884ea37caae2d20928f71ec07a74fc3a4bd3eec4708662a3ad4f4230ce57f1ffcff50"
    "3261865a920c82b637ca89bb20c3dce155c7ffbedf0afd5972331cc72c30f315f6f5b04c";

  // Convert hex string to byte array
  std::vector<uint8_t> d_bytes;
  for (size_t i = 0; i < d.length(); i += 2) {
    d_bytes.push_back((uint8_t)std::stoi(d.substr(i, 2), nullptr, 16));
  }
  std::vector<uint8_t> z_bytes;
  for (size_t i = 0; i < z.length(); i += 2) {
    z_bytes.push_back((uint8_t)std::stoi(z.substr(i, 2), nullptr, 16));
  }

  const uint8_t KYBER_K = 2; // ML-KEM-1024 uses k=4
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96; // Updated size for ML-KEM
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate d and z together (64 bytes total)
  cudaMalloc((void**)&d_d, 64); // 32 bytes for d, 32 bytes for z
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  // Copy d to first 32 bytes, leave next 32 bytes for z
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

  // Convert to hex string for comparison
  std::string result_ek = voidPtrToHexString((const std::byte*)h_ek.data(), ek_size);
  std::string result_dk = voidPtrToHexString((const std::byte*)h_dk.data(), dk_size);

  ASSERT_EQ(result_ek, ek) << "Encapsulation key mismatch";
  ASSERT_EQ(result_dk, dk) << "Decapsulation key mismatch";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

template <const uint8_t k>
__global__ void keygen_H_test_kernel(const uint8_t ek[384 * k + 32], uint8_t dk[32])
{
  H<k>(ek, dk);
}

TEST_F(KyberTest, H)
{
  const std::string ek =
    "eb7b84f8b29729015487434804a5b700d7239e222d8b14b3352421c5e4283e06ab1ca82375b00a1ba3c6d2ec81bcf60a7f55bf6d2796d1947f"
    "c79724195abc89f91e4b923c4bb268eb6caa56d42d061ac323697092e57f71333bc54bc46a97ace8510e320972de819eb357674545b07ea504"
    "a6f2799898201485a2039574272cce0dcb4167fc0830900ad4b8a6e46932c4791ad99c709a8046985c0ba43bc4be722be8e04ea9f13bdb49a6"
    "25e7cb226738b251cabf168136e7837d4cbc9d884a5b6b27eb1774bae78617b22f443423f2c1b369f402a8253d2dfa58821290a8b89849b243"
    "66c4be8626790c86b485067649d5b91ae9127fa7bd353963d85c02d7d53a1cf919363068c6c6ca88fcc115bcc6e3568497668d8a771bf97322"
    "e6e523988c4e25aacce93986c72a0481cca855a07ebec44f4ce08cbcb87d92e855fa541bf95975f3ebc7531c2e5e751e02f2c587f959546a27"
    "7fe2a7d778b75251cc6d47308d936a543683a49624e28320e693bca7bab37b5a9cc16367fcb1296f8063cbe92520a2c58f20b698079c4acc04"
    "4a2c3170768f0214b2e01726b5d18b2834bec28856084ac3a9c836b21196a952cb755759be50753a2902a8a769f41a4889b3af56bb63a8c88b"
    "1694aa7304b95ce0c691f6411f9905c9738e0a170a9a7babc2027f0493168952bae24c2a98b84af7b009c8091a8873bde88066f4dabbe65346"
    "78ba2d8818ce25fa9488b4cb49b3c006e47e773485f5b5107b589b89a686512b597d8816aaf4971e74914625585240a0cc2a2e85329efc38b5"
    "53441c39155eb1a9293f660b2fa5467f4362e8b6460a07217ea84638e016c9d95835aa8f8784bf26a1cb56a863f105807ccc6457b85c191320"
    "16280fd49a105ec9ca91877381801fd557c107ba78cc18093a1031860512e4a6cbdc37394f4319ad90cc6c548fea49cebad0797eda3aaf8743"
    "5b7b20b6d8b16d4bbee37bcfa4e1af1c2760a3fa1e45983b79e8654f880ae5a8aa6a414e42c9b9e41c82895214f06a1ab5729d1b5ab546783c"
    "46bb87c7fbabdf9c7672c8a5c2a1b43758807e882c61fc3c135b9cc7a044deee0ed0375fc79324512e51fed475f5ff6fbbe692008884ea37ca"
    "ae2d";
  const std::string expected_res = "20928f71ec07a74fc3a4bd3eec4708662a3ad4f4230ce57f1ffcff503261865a";

  const uint8_t KYBER_K = 2;
  // Convert hex strings to byte arrays
  uint8_t ek_bytes[384 * KYBER_K + 32];
  for (int i = 0; i < 384 * KYBER_K + 32; i++) {
    ek_bytes[i] = std::stoi(ek.substr(i * 2, 2), nullptr, 16);
  }

  // Allocate device memory
  uint8_t* d_ek;
  uint8_t* d_res; // Changed from uint64_t* to uint8_t* to match kernel signature
  cudaMalloc(&d_ek, 384 * KYBER_K + 32);
  cudaMalloc(&d_res, 32);

  // Copy input to device
  cudaMemcpy(d_ek, ek_bytes, 384 * KYBER_K + 32, cudaMemcpyHostToDevice);

  // Launch kernel
  keygen_H_test_kernel<KYBER_K><<<1, 32>>>(d_ek, d_res);
  cudaDeviceSynchronize(); // Add synchronization

  // Copy results back
  uint8_t output[32];
  cudaMemcpy(output, d_res, 32, cudaMemcpyDeviceToHost);

  // Convert output bytes to hex string
  std::stringstream ss;
  for (int i = 0; i < 32; i++) {
    ss << std::hex << std::setfill('0') << std::setw(2) << (int)output[i];
  }

  std::string result = ss.str();

  if (result != expected_res) {
    std::cout << "Hash mismatch:\n"; // Fixed error message
    std::cout << "Expected: " << expected_res << "\n";
    std::cout << "Got:      " << result << "\n";
  }

  ASSERT_EQ(result, expected_res) << "Hash value does not match expected"; // Better assertion

  // Cleanup
  cudaFree(d_ek);
  cudaFree(d_res);
}

TEST_F(KyberTest, ML_KEM_Internal_Keygen768)
{
  const std::string d = "d141fb6f9f887d58506a61956503ae1176235ae33a4cfefa6c537b16e795e744";
  const std::string z = "33c2a4cc415ee35af1c5b5a2e88436d5cdd737b6844ce1506d1d38394861f05e";
  const std::string ek =
    "d950cd099780a7bbb1f78b12cbc1455f72b06ddb8d56fc4042e272272241a0c654e73ba6d2906621cc8d65fc520f6c2144964f29d89b2b1806"
    "71da6a4d1c989311025d228912499afce881d8da044609b12d35036ed82eb8c79684d602d49c34993413c78462e65ba75f2333f5871da7d691"
    "d6089cde265db7f79be4987f8fc1ba3165295adc53454a79913151a1a69cba287725014b97dba89478295e126d58f881e2a49110850096c181"
    "a090808fc273715cb2c8c7506daabc3da5a6f1a39e847094ac47b487a81294faaeeab574692b3e7ba79c5eeb13370acf2900ba6f01611600be"
    "653592bd267ed148586269225bf89bedd38142939ee44b478f593aee93a8a77cc6e0b12b7d19cf2ce87752ab2fb57c7c2e846c1ff55d846783"
    "2218258772a328faa6ea4aa6f97b6bb8281ec245b5e6da9ef7133adf7660c9873335061a2d3347a9977f9ae5366ea21c2a7c7386d11f43ab9c"
    "57e0613f2c82239b85d093c8ed717ebc7b8cc6755d3579812a66860531336c91cb09eba2ec92c729a9946e0041d81a3a84a79f73880e97c229"
    "153b616b1142cc9c50922150edb53131645e02f7c447731994932568624c63e01a7b30c9750c94bcfccc2d0c3c5376ba87a505891c55262995"
    "f4e147b95a047f704db720bb3f5586ac9082fab95b3cb78772ba9634705316d0ac17faaf2a956981cc02af222e63a3a93c079fde158c3dd2b0"
    "21dace7626ad6f313ad951947ff6a6e0bc4d10baa06aa25f9acac990a535938aa6eada66601c1bc34b496c2887bf232587f51a5953a46ec518"
    "b8677f73e651e15b203b69888d35325fc538893a16d533c4a3289a8c195c64a83108d5bbcad51b6b2442c1ea0090e03b9d54a2e9d0abd71403"
    "070111805b2e9693b24e10466063bda3c557c75798290ac09ab1623dab64fffa43887b5766740559b3831f45bedfcc9033860744bc80eb4913"
    "4c153bc247b204478b57b999a14287a0c1ae0a9c693797bf6d16a9b328a19381028327255baa2ffcf452410ba07370100c744c11aaa1247c98"
    "5b433831e709e1947eb33b52f7b395e6d594e0b8200a4987326b68ffcc6e23d800d45440aa96446a294405c9469219519c2a70cc862af05384"
    "d9385dc55621935b7f10307a7fb90c17d02630a7185ad322fd8122ad78018692a3f3ac6cb8b6009d10ca24d0b700eacf20420041c7a905d8c2"
    "ee3590cc72ccbd6a9786c05abbc36ae8f141a690c21e2878c06b4727b2ccc0b73dbb4bccdcf7a70136123c3ba9cb27a583f372db25ce73825c"
    "dc50051dab07fbd539cd80131a7cac13231aef92ac1b963b5b02359f096d4086c52c2699104b2a7781741d0187d790af3231708b8cc2fb54b0"
    "b9d0b01d1c8181e56334466ef8088a5fda2004d24a821784da476427e3b43161bd3e5530502c6e1cb7a49268b7c58c83a63821c40b5102d029"
    "9d00a40f455e459018b056955a3aad88391ca78a10e203b427d4bf25197c5f166f47eb78e2f31b876b423a8c9d9e88acf6c66332acc93d9774"
    "9125c6da237969789aa7bc28f2451fb970ca2df219ec6c0adb1145571c751cf9ca2ae878793696abb2b1626a9e379315f9053b1ad26a1ba30d"
    "bfeb7b4c2c122b5004ee39bebcca00834dfd82f512ed3255e115c632c7aa6ca680727e98e8f2fd9486bfa152";
  const std::string dk =
    "46f52892e4182e356fb6970777515d3347cda429753c69a860cc7c24da3b256862e5e57a4d350add9009d83c0b4b104c1d654a586575dd5220"
    "335789dea3ba9cac0d7866ba5064920fd1325a839cf8a3b35246225a0c1ba7362e49e8742bec7e1699cd58383f02b9428d258ff28c0e764a39"
    "9888b70fdc0a7c65089ec64aa33b70d0b1c0d5c31bba3aa826a4476ecc4e445c2a1ea9b05402c265b8bf73537e57fc9f321706246b4789b051"
    "20e4b885cc7b442630d8451996b86cd075906cf475af07a0fcba10430ccdd0703eb5c34df0f0758c9b25d5b043cfc43845d937e6491a701046"
    "e5e515e2c28bab53139ba015e6db7851641b1e6c013e7aa16942011ec618836b8a9c45a0483a0af2e1664e86524051047e95988e4714326765"
    "95f46fe623260c07b58c42ce36357abb4b565450b49071cf617730cc002011836089f939ab3601c2c12082b74bee171b4dc2c8e4731d381b0c"
    "fadab78e3a0f984bb49342b894b25cdb6a273e49a27aba0bca0a0053e27cafca069cd157b4f2b2b933b619a28ac0fa87ddd1857c1189252a7f"
    "2160a3aeeacdfdbb9aac333dffa650ffdcbf3bf03775b9008a7abf80f3722840952e44aa776884b3f034dfba04fa25459d9aca6f182811e76a"
    "39bb7211aca3922415d4c34a52578bfe2ac80f09ce25c429cc771a52176c28ea5b0e65cd45a2a4f6b470bf6b17e1946620c48575217a796c60"
    "63e8ae32498984999f3b87b8de1b9eb2818f5fc5c112c04226c9c32e9b4871e994d9569084f31708aa261554a5f73b205096193b821cad7428"
    "3d7479f3a306c6796e31099a30500f84188591d3027ec0836665a972307afcb6a14261751a551cbfb67ecbcbbf3593b6e6e61fe58412a3a6b6"
    "6133b8fb2753090004028661ad5cc41192686eab1df15c7fec94235a00b1f74c9d5a55b9d7b0a2d14b2ebc1873e2e264bd08d0f76a0d62c4b0"
    "8d663f8e9330576148bb78255fc23832aab69b13b9a9427af5e75f439472d51126c53a55caf3616da07e7bc2cb6841959755065959c8486a27"
    "01f04d576799c3ca9af8f8427ed255882653fbd8118d0a440885c50b474ec4f217f981630746188ba8aa22684ccbb6cab0a16da0b586677892"
    "a95b4c9cb053b26398b3955654576cbaf72fff9b962059b80d34a09b133b681b2898288e155a29f507a48822827bb02baae4967c5527ab634a"
    "3dec781dd351e4c45f1b6bb019f56adb7039a986873ba08c5ee47903a3529e86a739b495fc41618a8a60df7819bfc93882d320b1a23748b132"
    "c35911c9905f6ff117717629d60c243af42cc6150a1489277af23c6ed5479c0c6434a1bf2b9713e61c7ed15314ac0c7ff1167a42e6ac27f27e"
    "864210ade92dc9574823234fd9da062a307f0cf6b4c9e44a60e9bf13da3b4d32b55288448fa68daff54ae4dc835739bd9089731a9219668951"
    "e0462bdbab0db4eba994283d36e3652fc184b765a46c0925b9f71757565407d89f8117c515228f28876ef3d0a41581bc5e19a30986b64276bf"
    "35c9246f9483a9710a579a399bc485f3576d47d30c78a4c4586941a7a1ac36d340e34774ccc398ff704ad19395a459338ad48873b5728c1596"
    "3c28a4ba31023148aa477686d950cd099780a7bbb1f78b12cbc1455f72b06ddb8d56fc4042e272272241a0c654e73ba6d2906621cc8d65fc52"
    "0f6c2144964f29d89b2b180671da6a4d1c989311025d228912499afce881d8da044609b12d35036ed82eb8c79684d602d49c34993413c78462"
    "e65ba75f2333f5871da7d691d6089cde265db7f79be4987f8fc1ba3165295adc53454a79913151a1a69cba287725014b97dba89478295e126d"
    "58f881e2a49110850096c181a090808fc273715cb2c8c7506daabc3da5a6f1a39e847094ac47b487a81294faaeeab574692b3e7ba79c5eeb13"
    "370acf2900ba6f01611600be653592bd267ed148586269225bf89bedd38142939ee44b478f593aee93a8a77cc6e0b12b7d19cf2ce87752ab2f"
    "b57c7c2e846c1ff55d8467832218258772a328faa6ea4aa6f97b6bb8281ec245b5e6da9ef7133adf7660c9873335061a2d3347a9977f9ae536"
    "6ea21c2a7c7386d11f43ab9c57e0613f2c82239b85d093c8ed717ebc7b8cc6755d3579812a66860531336c91cb09eba2ec92c729a9946e0041"
    "d81a3a84a79f73880e97c229153b616b1142cc9c50922150edb53131645e02f7c447731994932568624c63e01a7b30c9750c94bcfccc2d0c3c"
    "5376ba87a505891c55262995f4e147b95a047f704db720bb3f5586ac9082fab95b3cb78772ba9634705316d0ac17faaf2a956981cc02af222e"
    "63a3a93c079fde158c3dd2b021dace7626ad6f313ad951947ff6a6e0bc4d10baa06aa25f9acac990a535938aa6eada66601c1bc34b496c2887"
    "bf232587f51a5953a46ec518b8677f73e651e15b203b69888d35325fc538893a16d533c4a3289a8c195c64a83108d5bbcad51b6b2442c1ea00"
    "90e03b9d54a2e9d0abd71403070111805b2e9693b24e10466063bda3c557c75798290ac09ab1623dab64fffa43887b5766740559b3831f45be"
    "dfcc9033860744bc80eb49134c153bc247b204478b57b999a14287a0c1ae0a9c693797bf6d16a9b328a19381028327255baa2ffcf452410ba0"
    "7370100c744c11aaa1247c985b433831e709e1947eb33b52f7b395e6d594e0b8200a4987326b68ffcc6e23d800d45440aa96446a294405c946"
    "9219519c2a70cc862af05384d9385dc55621935b7f10307a7fb90c17d02630a7185ad322fd8122ad78018692a3f3ac6cb8b6009d10ca24d0b7"
    "00eacf20420041c7a905d8c2ee3590cc72ccbd6a9786c05abbc36ae8f141a690c21e2878c06b4727b2ccc0b73dbb4bccdcf7a70136123c3ba9"
    "cb27a583f372db25ce73825cdc50051dab07fbd539cd80131a7cac13231aef92ac1b963b5b02359f096d4086c52c2699104b2a7781741d0187"
    "d790af3231708b8cc2fb54b0b9d0b01d1c8181e56334466ef8088a5fda2004d24a821784da476427e3b43161bd3e5530502c6e1cb7a49268b7"
    "c58c83a63821c40b5102d0299d00a40f455e459018b056955a3aad88391ca78a10e203b427d4bf25197c5f166f47eb78e2f31b876b423a8c9d"
    "9e88acf6c66332acc93d97749125c6da237969789aa7bc28f2451fb970ca2df219ec6c0adb1145571c751cf9ca2ae878793696abb2b1626a9e"
    "379315f9053b1ad26a1ba30dbfeb7b4c2c122b5004ee39bebcca00834dfd82f512ed3255e115c632c7aa6ca680727e98e8f2fd9486bfa152d4"
    "58060c25ab38d797c7eb8ab0b781bbfaa914cccc5ac14818fc8850202e722d33c2a4cc415ee35af1c5b5a2e88436d5cdd737b6844ce1506d1d"
    "38394861f05e";

  // Convert hex string to byte array
  std::vector<uint8_t> d_bytes;
  for (size_t i = 0; i < d.length(); i += 2) {
    d_bytes.push_back((uint8_t)std::stoi(d.substr(i, 2), nullptr, 16));
  }
  std::vector<uint8_t> z_bytes;
  for (size_t i = 0; i < z.length(); i += 2) {
    z_bytes.push_back((uint8_t)std::stoi(z.substr(i, 2), nullptr, 16));
  }

  const uint8_t KYBER_K = 3; // ML-KEM-1024 uses k=4
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96; // Updated size for ML-KEM
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate d and z together (64 bytes total)
  cudaMalloc((void**)&d_d, 64); // 32 bytes for d, 32 bytes for z
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  // Copy d to first 32 bytes, leave next 32 bytes for z
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

  // Convert to hex string for comparison
  std::string result_ek = voidPtrToHexString((const std::byte*)h_ek.data(), ek_size);
  std::string result_dk = voidPtrToHexString((const std::byte*)h_dk.data(), dk_size);

  ASSERT_EQ(result_ek, ek) << "Encapsulation key mismatch";
  ASSERT_EQ(result_dk, dk) << "Decapsulation key mismatch";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, ML_KEM_Internal_Keygen1024)
{
  const std::string d = "7c60c04b7b29b64d517308ff685f3fbfe4585fe4de2bbe17b50c98861455e09b";
  const std::string z = "a97c2d818858c4047088fa19dfd5bd6ad4300c7da9c3948c271da06adc2c730c";
  const std::string ek =
    "01a9925d608f9a4b07db948c1240c712ecc49662890f70b45a493dd0a240e2546ca4b52a677c5fdbb13dbe4a2bc75b901e7ab81ea21d868624"
    "1bb24d29507d71a83dd1c7b1099710ca374acbb4c027f79cb3bb765627661af9be0185b03259c5041597a8c10414b01cd96262398343bf697a"
    "8fc4ba7798a7fdfacf64480a9dec4f314b606d2851ac19227c661bd3ba47f7e6a5b8c51cccec7fd9562402b40c563bada34625a1a65ee694b9"
    "825907d32679f871ae05129fa2d348b190a78f456915247016c8023b293cff3860268c5c743873704bc81de143fdbb47ee104b71b529871622"
    "ddf8342205a2145005a4d279b37a8a5fc73f29bcb910c2aea0c81e95dc51dd28b46e910665910c6a4a6659bbbaa82b9b1972849b4a564f6460"
    "ee2ca7df144bc3d8a04ae441dc734e94a0c163d010ffdb35a8a05b923983fabc593ee703ab1082ea768b81733b315a63af1a85f4f89a89f652"
    "8c4390f4731b2fc3c80aa711c11910fae6b49e63b3fc0258d2e33d7abb1daf920d31855db70c3ee84ba33a5bba859bbd0f3972f0a7a599383c"
    "5f25393dec3d3a511873c99386c666995702ca7a2b2ada5ae0d59a37a07e924568307161794739b8eb3a289035e672c93b2317b7f37d3cb43c"
    "890a9729839c453552dd615a049959003482cbba467dd747510c1e5f247ddffaaccc6b131300b6bb599d42522c1dfb3e77021c130181934961"
    "e0b1a00a006957f1902689646de8b3f3339d64ea9a9b17999b14995bb89637a466b59537e2611aba87a7e447b58dd161baca731164a9c8cc66"
    "e7bacc9eb16d5bb32e5094a8f2d35768616c5ebb5b7d2aa9be015695857ad77812168959cc887e7c9c35efca40e67703f4336a3b379b6e069e"
    "77120a977033307a3c0981223541102352b2a9888feac1a80c7563da6651e6bb29b2f25e84c8425758aac3080311259210358f010132414251"
    "06cc090594b4867654fc5569b39ab2d2105f1338263e7acaeb199148aa018e820c81194bc3904f7d17663fa407d53cb657e0a7880032d09642"
    "4f0c1efa8c9fa28737d8196b06171b59e050ba17b9322c08c4546b524a644c85b23496a44578397e67c2226081847809f2dc48bfd28f99b89a"
    "8efc7e7b54216b885485625c001a2604a22ddbca9e8e36b3c3f379c01538b01bcf3184406f87cdcb37488205896c5c5152f2ce59b784820723"
    "55911be5a5cf1ca529a0cc05c4c4a6c1e0a4a468b15e7cb2c255a23a7576b7fa909a0a2b3f106708259ab82c3cf87b1f96330b391654c7468c"
    "2b1789c443c59dbbbbfb04a7e6627196b2159f7619cc87b6aa876f724187b78aa9c4e484b3500d90e5b5d08732a0e704d09183ed278fc4135f"
    "b21212f21884cc5aa2b173875ffa4b9aa26ed6f8232e11558b4956e62c6711410f88aaa4352a39705094aeb461bfa78866c6674dd652420527"
    "80dabbe39b8d3cd50d866acdc6976fd7d5291e39cec9f51e32930c52607950446da41b0767208ec746831608885af286522461acf702488b85"
    "bbe447b3b4c811f4a589997c5398889ab36f679b4bc13316d96aa6bf2a60cf854932e62bf10240bf38347e023780cba046226d85184d15d077"
    "c153540518c71267a2478624ec684300bb430ac6b5db056bba889c6784440cb1ab9b51995e368e0c3a32e4c9c85c144e87352fc7bba9feb8b9"
    "350a89ae815114713e6fc72120d1579de3c2cd518401114b81c97d763207026b6203eb13243cae3e6cc0fbe10a05a56eb9009481a002effc2d"
    "c377418e9015f24c2c5f1b0ca0f98008a676ecdc07ec810eb1d41d6600caeda3c2cb908b24f9632ab186d9f3177284447aa785e7e80f647c1b"
    "b8194e3f2b9f1c00a207b28910cc58feea13c929a9c61536ff532d52f4862d0276c46609428c370bea5b41d3396361585971b8cc77b8a67b0d"
    "fb64070078764d30872cd1b455db171b03b409376274a22f251a913b27bc60fa97f266239d3c1d85f19d8318c1d4a400bbdc66b2207a314275"
    "cbc8281a7264c31336ea47761a266160a08bc984bba61011bdeb8fd2cbc15ae53c423131f5441c05e3787620ba90493e8bcc814485a5723ab4"
    "45777598b31ab1b3bbcc510865625c71b56d55f23ad208b90af206c91740076524656814edca3f8b1c99b3d69418955dba345f30601f238fa2"
    "ac10c2eb9e393b3da0b363653d1091c819c185df20b53c0e5312c4284f";
  const std::string dk =
    "5ffb1527d2a174dc127cf5ad1243999e2c2e4f335b453b8ecd96204486692a8cb83909c6598207514634b8f727e9c05a352326fb80cf15f2c8"
    "fe6c6cab0a3dbd83bd54e535bde469154bb8d06cc2908a20c36079fee961ef08068415145324bd6c5938e899cac7717642f84dee51b4fcfb58"
    "a67729d9b2bed209237adccb7f1a5bd0a40eba607e4d77ab5bfa57a0e5b44daa39179c6b28225d57dc6fdcd1adcb29b53af586f2451317abbd"
    "b0ecac42a4c6b5c47c20b5798501cd0e4ab0a86913d435049aa64ec34189bd6ab8ccb509cd7471e5b34a3d811e4c8c90b216ba225213577294"
    "18eb1d9e0c90dc319f44f2b00bb8b612f207ffc0b832e09170f25136717c201515c6fc8b42ba6d5d85c56f0b4c083ca1878b6c70d00f2b71c8"
    "26458c9754cfe021096ca51753cc6f7182577ac40c6965b92e72465d28c47bb42e30771762104510cccdcc3975b87082c6379983a1a99c715d"
    "24e25b36f2ce485a5823e8b8dffb0c13aca4edb13fe6c1861716595ed1bdf3aa917c3491e2f00d42a25e8c0b0ddf331f76e2264eeb244f014b"
    "76e125192475e255be8b68baff6ba058910589b73ca200323ceb3e6b588b8f26909d737c3b633e7c71b74db03bd3a44d6ae287dd60bb58ab37"
    "da25171dd4676ad7b410ba07be649d2b928db2c8008b5765b3b03fe53bb28a25b7dc2a278e4b5b9bec125ae813c4099c29c40124b2845ec804"
    "0442cd74e194c7a703def73f3026225672c860c1a8eda55c44b72f21515dfe95b602432cd2ecc5be4385ec92a1603a98a9318cd74270051076"
    "5c6a563ef4088c54cca6ea22a4b311efaa385dc6714ee9be55c147fbd791eccc6cc1d29acbc91f837ba236921626b35cc4fb0bdce0a77535b0"
    "748b0a9edc98e5754073ab27f5164f03f2814f092e950c0f3c9935e28b1d23a3c20c367d1127a722a42521385157588f16aa1272966b2cf083"
    "397697ff099f18107cf7a9972b8c5b7d3420b5c67b8a61aa8ae33b4352ba580c7c2bc20e9acc1c6e430e10d90525a746d1981166738bd2c202"
    "48816dd077b6ea08b954691c6740a3fd79c640b02dfbf86a4ae19d0cb01033421ad92944b68c1dc2bc135333039b585d97929c90a7c117c0b8"
    "e6667c29421e4b2630ea2ca7a0e7cf9ae10a1e25a9ed62cae29c6e03fba30777b8e2922db8d0b6e598bc618956cef5648df3ccff8a428d9c42"
    "6a24bddcb74dc2b19c3f208782a06517599c7b486ec8b99119f18940045463ab6e7d9cb5a04961b9f38d817a24f4d8aefa006b94d1acba5c18"
    "27e2998d235eb510038e8556788a876f4406d3567d21326f47c532aa4201fc846f7098b35ff733c53958dfb3bd642bbb82e59a9ec937fa9a27"
    "51856223399af7382b6e677bb9a8603a9663bca44528257e75f99f51577754f57a7c2b88a392b167026c3fe13121c2a8bf133208a5919e188e"
    "6c861122e3be1dd536cc2773b50879739306d03612b9472ca0ac77c1624597f35862901685b80433347824785c76b30e0a375827047e5af566"
    "2b405c1b9a9aee926726617d46c6121b27c212650eb4c202a96941dbd3a8ce562a8ce86732f1078d2a46386276b8d219ace577c7ec4310c715"
    "9f31af569397e0fa7648b7cea6095653d21ba5f2558da5a483db0ca2db65d6137796f27944891960c9659bbca4fa7960abc8bde620225729c1"
    "66a89eef1c12c7609cfda46e3381ac6b1840e54c792711664bb44fa576a58ec2af4ee4b7fb1342a2bb4d8f8533411cbc28064350112399a127"
    "95095c7ccab1e5ab948221a1968a84d675311d747ccb64acdc871bb5e2237bbc56aeb2a04ddc3f853b0218fcc253e3b0b78c5824506420ec6b"
    "e687287ec2540392264c27550920a701333b679b45f5354244f6ae74b8788961c5382b5ec042c1ecc3be11e9a543f180596b15109b759a1a37"
    "a3015afd56843d378a24eb5f90c33ae9d5411ed38aad95a5c88847727c70d4c72ce0d296509a67da10434b334f365794fdcab01582c4d5570f"
    "e0cb0286489b56258b9e07905f040926c030b46a5ea7988c19118567c40b0bd09b85284c34745a794a31e910762260be58101115bc8539e740"
    "20ba8f601c2fe4555daac4c2fa60b99899cc8e042c45636baec66229fb2d48bbbd4f12462e278656319252a60eb8034ae4f90c43b79e01a992"
    "5d608f9a4b07db948c1240c712ecc49662890f70b45a493dd0a240e2546ca4b52a677c5fdbb13dbe4a2bc75b901e7ab81ea21d8686241bb24d"
    "29507d71a83dd1c7b1099710ca374acbb4c027f79cb3bb765627661af9be0185b03259c5041597a8c10414b01cd96262398343bf697a8fc4ba"
    "7798a7fdfacf64480a9dec4f314b606d2851ac19227c661bd3ba47f7e6a5b8c51cccec7fd9562402b40c563bada34625a1a65ee694b9825907"
    "d32679f871ae05129fa2d348b190a78f456915247016c8023b293cff3860268c5c743873704bc81de143fdbb47ee104b71b529871622ddf834"
    "2205a2145005a4d279b37a8a5fc73f29bcb910c2aea0c81e95dc51dd28b46e910665910c6a4a6659bbbaa82b9b1972849b4a564f6460ee2ca7"
    "df144bc3d8a04ae441dc734e94a0c163d010ffdb35a8a05b923983fabc593ee703ab1082ea768b81733b315a63af1a85f4f89a89f6528c4390"
    "f4731b2fc3c80aa711c11910fae6b49e63b3fc0258d2e33d7abb1daf920d31855db70c3ee84ba33a5bba859bbd0f3972f0a7a599383c5f2539"
    "3dec3d3a511873c99386c666995702ca7a2b2ada5ae0d59a37a07e924568307161794739b8eb3a289035e672c93b2317b7f37d3cb43c890a97"
    "29839c453552dd615a049959003482cbba467dd747510c1e5f247ddffaaccc6b131300b6bb599d42522c1dfb3e77021c130181934961e0b1a0"
    "0a006957f1902689646de8b3f3339d64ea9a9b17999b14995bb89637a466b59537e2611aba87a7e447b58dd161baca731164a9c8cc66e7bacc"
    "9eb16d5bb32e5094a8f2d35768616c5ebb5b7d2aa9be015695857ad77812168959cc887e7c9c35efca40e67703f4336a3b379b6e069e77120a"
    "977033307a3c0981223541102352b2a9888feac1a80c7563da6651e6bb29b2f25e84c8425758aac3080311259210358f01013241425106cc09"
    "0594b4867654fc5569b39ab2d2105f1338263e7acaeb199148aa018e820c81194bc3904f7d17663fa407d53cb657e0a7880032d096424f0c1e"
    "fa8c9fa28737d8196b06171b59e050ba17b9322c08c4546b524a644c85b23496a44578397e67c2226081847809f2dc48bfd28f99b89a8efc7e"
    "7b54216b885485625c001a2604a22ddbca9e8e36b3c3f379c01538b01bcf3184406f87cdcb37488205896c5c5152f2ce59b78482072355911b"
    "e5a5cf1ca529a0cc05c4c4a6c1e0a4a468b15e7cb2c255a23a7576b7fa909a0a2b3f106708259ab82c3cf87b1f96330b391654c7468c2b1789"
    "c443c59dbbbbfb04a7e6627196b2159f7619cc87b6aa876f724187b78aa9c4e484b3500d90e5b5d08732a0e704d09183ed278fc4135fb21212"
    "f21884cc5aa2b173875ffa4b9aa26ed6f8232e11558b4956e62c6711410f88aaa4352a39705094aeb461bfa78866c6674dd65242052780dabb"
    "e39b8d3cd50d866acdc6976fd7d5291e39cec9f51e32930c52607950446da41b0767208ec746831608885af286522461acf702488b85bbe447"
    "b3b4c811f4a589997c5398889ab36f679b4bc13316d96aa6bf2a60cf854932e62bf10240bf38347e023780cba046226d85184d15d077c15354"
    "0518c71267a2478624ec684300bb430ac6b5db056bba889c6784440cb1ab9b51995e368e0c3a32e4c9c85c144e87352fc7bba9feb8b9350a89"
    "ae815114713e6fc72120d1579de3c2cd518401114b81c97d763207026b6203eb13243cae3e6cc0fbe10a05a56eb9009481a002effc2dc37741"
    "8e9015f24c2c5f1b0ca0f98008a676ecdc07ec810eb1d41d6600caeda3c2cb908b24f9632ab186d9f3177284447aa785e7e80f647c1bb8194e"
    "3f2b9f1c00a207b28910cc58feea13c929a9c61536ff532d52f4862d0276c46609428c370bea5b41d3396361585971b8cc77b8a67b0dfb6407"
    "0078764d30872cd1b455db171b03b409376274a22f251a913b27bc60fa97f266239d3c1d85f19d8318c1d4a400bbdc66b2207a314275cbc828"
    "1a7264c31336ea47761a266160a08bc984bba61011bdeb8fd2cbc15ae53c423131f5441c05e3787620ba90493e8bcc814485a5723ab4457775"
    "98b31ab1b3bbcc510865625c71b56d55f23ad208b90af206c91740076524656814edca3f8b1c99b3d69418955dba345f30601f238fa2ac10c2"
    "eb9e393b3da0b363653d1091c819c185df20b53c0e5312c4284f4f61fdb97b19bd7e5ecfa00dcf622b50c454288679d71d0b18b5a2b34a1d93"
    "15a97c2d818858c4047088fa19dfd5bd6ad4300c7da9c3948c271da06adc2c730c";

  // Convert hex string to byte array
  std::vector<uint8_t> d_bytes;
  for (size_t i = 0; i < d.length(); i += 2) {
    d_bytes.push_back((uint8_t)std::stoi(d.substr(i, 2), nullptr, 16));
  }
  std::vector<uint8_t> z_bytes;
  for (size_t i = 0; i < z.length(); i += 2) {
    z_bytes.push_back((uint8_t)std::stoi(z.substr(i, 2), nullptr, 16));
  }

  const uint8_t KYBER_K = 4; // ML-KEM-1024 uses k=4
  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96; // Updated size for ML-KEM
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate d and z together (64 bytes total)
  cudaMalloc((void**)&d_d, 64); // 32 bytes for d, 32 bytes for z
  cudaMalloc((void**)&d_ek, ek_size);
  cudaMalloc((void**)&d_dk, dk_size);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq));

  // Copy d to first 32 bytes, leave next 32 bytes for z
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

  // Convert to hex string for comparison
  std::string result_ek = voidPtrToHexString((const std::byte*)h_ek.data(), ek_size);
  std::string result_dk = voidPtrToHexString((const std::byte*)h_dk.data(), dk_size);

  ASSERT_EQ(result_ek, ek) << "Encapsulation key mismatch";
  ASSERT_EQ(result_dk, dk) << "Decapsulation key mismatch";

  cudaFree(d_d);
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, H768)
{
  // const std::string ek =
  // "3538a151f682d21a5b15e8918f98cc178976893aa3ec70807e4809134c63eea79aed92044c2553976a2f7b93c196c54711b341516ba53bfb85c3d6872358258b4b18871049f5776977860e81945d2eaa6bd8714d894474a93024771a25195808b1c889c313b662962ac538b0bebb79e25985b201199a212137f1be9780144488211216c40b404148b98e6d4082770abd62f55679a1559f7092c1763a8b2cce50b87480bac89edb4072525651c4753075460a1cc6c9d23ee394b98f645039d3c87f1b9cfc3097108bc15ca33f3c818a3937531744772eec0223e341b50a71e81b76d2b81b2b895cd5550a75061c8d47645a63504f8a32aa724cd2f495afc9bb9a655ecb308722d68239a37be886900d315d77148d67e2778ba22891717292773c3d158663a0939f111495a14df9e74d062263a4a08baac89ebe76c561c0c3f124377be314bd6047c6aa2ad2ec70ea834f1ea16339f278600398df240b868079ee7621fa43751b7888e883862dc8262160ab4344b6327a587cb569dbb0b414772a2c1b89d009c20ffa2fd9f1a027a57fac631005c46aa0061cc8d7b2fa0807ae056e8de368b3421945f988e1b6c4add117584b9ad9e90e9b960b26abb8e5c249e36275d7467abfec09b75ab95e9b5d61270f77699a0bf206253a5026f03c7c1454d131b2bbe252f1e545ae45329e279eda90581ec96096710fa0e8855366cc9e9191662b6dfde957ca5764fe0aad6c7911ce8756bc05a78fe616f3b5c08917bc6da485653410c7f345c4f80a132442de3a6fb0cb9a46837a19d3bad6ab9e42c507080271716c8cc2bba94a18ccf4cb5a28297a8d1345174a6c27f6464f31c92b3474906ac9b1059c2eccca1e99cdd255cbe90776c47bb41f494580b8544d3033bd1b6f5a5778faa17656538430b92ed1a1930614821d946dbb45b79fc3cf16d266c6d6b062934efdecbc19544cbcf541e9204859ec0ced2289a93b53552399b689182242c48ad1cf25f2be2a6a46eabaa733cab7e398106470a2f787a0e1405dd28233ad480f30bb9e9ac3c260f466fa267b238b7b571c7ee9d3c81ef38d8920772ac47417c4aaaaac3922c3bf299ba43ac02bca745c7d4c78e5d2ad38c45f27104f88f498dce26463357892b0b5b694afaef5aecbf466ee32114dea28e37a9b949265be7878392682314c37399251d7e81c23840b888a2063d290854c3a7c3269c9027b36930e7c13b76f393a75082286ab13713255f5b59e70478fa8510833625f1bacceeda3b080f88f8959a8fb4043353632fb748ab90297a01a54a9c81e5d11c7a8e6b3c6300c1a0a0d203c4e16ca32f9dc2dfa044b4b1b9535d73983788d0356522cb483bf00c219669075c145a9a9bf11b99de2372fba6811363a1dc36a263f3589cf8b3442877d17f017bac84cf89b2ed5c81e15f2b3db89b470811eb2073fb8f0605624998fa0854e848c70e901946c1eb433754d98ae31f04d79b37be6858e659b6575981c489c936864a23a2c452734815f134c019b267846aece30b9d13b5edb569d98e9643a1b9b057b9118d4004363be3a01aa14875e74c54a4f116d8cd4c39cc58d904565f39722e15242905b7f3a94388ac71804a2a8abe1567349206dbca397032f03c99b15489f7b1d75839f2600bf5d41c2be1fb8f176be";
  // const std::string expected_res = "005d2c932d1a7c4e494af7ecdbd6fd320d11b4def511c44a9f67bb6604444c85";
  const std::string ek =
    "c9e226d9f303ca090c39e64f3273b04ee124ee2966eb8a176a205931055881927885a4047e5a0cc1954ce22c96ca76ba6a539333e24a1aa34e"
    "a11976efb4127c512ab7693343851caf2c5f4449a524b872a1a62cf3a24e8d69149ae8059114332df87ffd0c3f1e29885dfa04a1d1925dd513"
    "8cbc71ed2b5a7a564f7e90431c7336c829802c96991688924efa8ed4243ec2e4cd1e20cb83d719be292971a15effd9cba582b7fd1087175508"
    "90761ca006a5bf065a65984cd669109320117ff12eb8828fb66324f9042d65908310369f378a2812352a9ae58d80f68389d43c848451024bc1"
    "1c392ec642b1044c990aeb25d2aa070385996c5b7d19504c28a19462d49bc9c4839d1772e3c795f1cb6d521b73a980a6ff1c9b4ef53197c01e"
    "d6e076e6a27ca5fa9811ccb19da7809237b6b3314660d5257d747315d46abd835c50ea173ef9196eca574c309f100c84ae94605374bee0eaad"
    "a08c7c2e9254fa77c0485a2be823c622700d525b873504ab552396d73324764483480659e1a9442342008ae089cc53c0a0f7485c4191326405"
    "fb484ba5d0b2162b0639d8617ecbc43cacacba33225f912560303d05662632d48b38552634670dee3539a911528084af5a1a448dbbadbb7432"
    "2dd7b0a3a576e1b514b380695a3835aa1c247e6526df04cadc92139520b274e3697678117f6c1cac4664ddd93b8f31441180c80e9c7cc0987e"
    "471ccf6e7811c3f5c801586b04eaa2f391bf88857acba0a4cee522e4e23735ca41f7d794ca9259dac67a7ef9bc9fc094a84059603298291a77"
    "46aa595e37317e22497fec7d06db123579b2ede67547c53db2662c410c3e76ccb278d75152a0af5b5b5f0a7480aaf01a2af4ce2dd644836b91"
    "033a4dbabb7926c84822d44a67b9c225e53a3aac003b308e3e04a3da3b8412c09eabb33558730f15c841df075375238dc61a9ff0fc2e372514"
    "d8353b40dc5964ccb41e8b1ceee10866549af3d0cd64ac4a643707e4166dfa348ae0b04c03847e29125b3c7592616baf3124bdccfc00275953"
    "c98a1b670c6d7ca211564cac64727dd3991442105832885bab298afc053562637725819f5706938613b42ceb4107c739bc5c8a4de7aed6670e"
    "2f717b73ea4be4492d320c5447a03fc22aa712d2054a90c19507261f3864f0407b54c92e3b1b15869b3cf3229c9c6c0f754783c4a3867e071e"
    "6199165b41188c641fdb7b5f56105e9bd85c1a99953e2881c2461aed731595bab6df29493df53060aaba7a517716b0a63ae5798a470d9d502e"
    "8af94f533623fb03a0277514d1fb637c7c5123444e765bcc83e9211e22b6771a83b0d129d1978ac03815e241215f2a7385f097458056c775b7"
    "4284c03c5439b5055e2cd84fa3e004e01047c46c6169002b7f3926ee69984c3c2d90c68ada8a0308137ea13715c7d8077b5017b6281fcc8c61"
    "bb181fcf659474b331e474444c2366d25c154ea255fd7c07a825c4a74834aa30c55e31bb8dfa4ed82b905c2cc4eb5a0944949072ea5c224469"
    "5ae4ca9f81a20e2539f8f183f899c6f5701d2d25b8ff626419e252a4459cb9ab8c3ba8b484377ea2a2bc518222a42131e992cd6eaa2c863060"
    "eb4885c067ab3cfc1bff036d9e464209a50d33525ceaaf2e2e901f1523db6b76c2ae26299c21ab87ab16618f";
  const std::string expected_res = "3db70d1db87d3509027488ccf0147d28e0d72621a4b2b495f58440ea96f3788c";

  const uint8_t KYBER_K = 3;
  // Convert hex strings to byte arrays
  uint8_t ek_bytes[384 * KYBER_K + 32];
  for (int i = 0; i < 384 * KYBER_K + 32; i++) {
    ek_bytes[i] = std::stoi(ek.substr(i * 2, 2), nullptr, 16);
  }

  // Allocate device memory
  uint8_t* d_ek;
  uint8_t* d_res; // Changed from uint64_t* to uint8_t* to match kernel signature
  cudaMalloc(&d_ek, 384 * KYBER_K + 32);
  cudaMalloc(&d_res, 32);

  // Copy input to device
  cudaMemcpy(d_ek, ek_bytes, 384 * KYBER_K + 32, cudaMemcpyHostToDevice);

  // Launch kernel
  keygen_H_test_kernel<KYBER_K><<<1, 32>>>(d_ek, d_res);
  cudaDeviceSynchronize(); // Add synchronization

  // Copy results back
  uint8_t output[32];
  cudaMemcpy(output, d_res, 32, cudaMemcpyDeviceToHost);

  // Convert output bytes to hex string
  std::stringstream ss;
  for (int i = 0; i < 32; i++) {
    ss << std::hex << std::setfill('0') << std::setw(2) << (int)output[i];
  }

  std::string result = ss.str();

  if (result != expected_res) {
    std::cout << "Hash mismatch:\n"; // Fixed error message
    std::cout << "Expected: " << expected_res << "\n";
    std::cout << "Got:      " << result << "\n";
  }

  ASSERT_EQ(result, expected_res) << "Hash value does not match expected"; // Better assertion

  // Cleanup
  cudaFree(d_ek);
  cudaFree(d_res);
}

TEST_F(KyberTest, ML_KEM_Keygen768Batch)
{
  const uint64_t batch = 2048 * 4;

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const uint8_t KYBER_K = 3;
  const uint8_t eta1 = 2;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96; // Increased for ML-KEM
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate memory for batch
  cudaMalloc((void**)&d_d, 64 * batch); // 32 bytes for d, 32 bytes for z
  cudaMalloc((void**)&d_ek, ek_size * batch);
  cudaMalloc((void**)&d_dk, dk_size * batch);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq) * batch);

  // Generate random input data
  uint8_t* h_d = new uint8_t[64 * batch];
  for (uint64_t i = 0; i < 64 * batch; i++) {
    h_d[i] = rand() % 256;
  }
  cudaMemcpy(d_d, h_d, 64 * batch, cudaMemcpyHostToDevice);

  // Launch kernel for batch
  ml_kem_keygen_kernel<KYBER_K, eta1><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Cleanup
  delete[] h_d;
  cudaFree(d_d); // This frees both d and z
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, ML_KEM_Keygen512Batch)
{
  const uint64_t batch = 2048 * 4;

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const uint8_t KYBER_K = 2;
  const uint8_t eta1 = 3;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96; // Increased for ML-KEM
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate memory for batch
  cudaMalloc((void**)&d_d, 64 * batch); // 32 bytes for d, 32 bytes for z
  cudaMalloc((void**)&d_ek, ek_size * batch);
  cudaMalloc((void**)&d_dk, dk_size * batch);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq) * batch);

  // Generate random input data
  uint8_t* h_d = new uint8_t[64 * batch];
  for (uint64_t i = 0; i < 64 * batch; i++) {
    h_d[i] = rand() % 256;
  }
  cudaMemcpy(d_d, h_d, 64 * batch, cudaMemcpyHostToDevice);

  // Launch kernel for batch
  ml_kem_keygen_kernel<KYBER_K, eta1><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Cleanup
  delete[] h_d;
  cudaFree(d_d); // This frees both d and z
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}

TEST_F(KyberTest, ML_KEM_Keygen1024Batch)
{
  const uint64_t batch = 2048 * 4;

  // Allocate device memory
  uint8_t* d_d;
  uint8_t* d_ek;
  uint8_t* d_dk;
  Zq* d_A;
  const uint8_t KYBER_K = 4;
  const uint8_t eta1 = 2;
  const size_t ek_size = 384 * KYBER_K + 32;
  const size_t dk_size = 768 * KYBER_K + 96; // Increased for ML-KEM
  const size_t A_size = 256 * KYBER_K * KYBER_K;

  // Allocate memory for batch
  cudaMalloc((void**)&d_d, 64 * batch); // 32 bytes for d, 32 bytes for z
  cudaMalloc((void**)&d_ek, ek_size * batch);
  cudaMalloc((void**)&d_dk, dk_size * batch);
  cudaMalloc((void**)&d_A, A_size * sizeof(Zq) * batch);

  // Generate random input data
  uint8_t* h_d = new uint8_t[64 * batch];
  for (uint64_t i = 0; i < 64 * batch; i++) {
    h_d[i] = rand() % 256;
  }
  cudaMemcpy(d_d, h_d, 64 * batch, cudaMemcpyHostToDevice);

  // Launch kernel for batch
  ml_kem_keygen_kernel<KYBER_K, eta1><<<batch, 128>>>(d_d, d_ek, d_dk, d_A);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Cleanup
  delete[] h_d;
  cudaFree(d_d); // This frees both d and z
  cudaFree(d_ek);
  cudaFree(d_dk);
  cudaFree(d_A);
}