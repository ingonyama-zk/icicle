#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"

// using namespace field_config;
using namespace icicle;

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

static bool VERBOSE = true;
static inline std::string s_main_target;
static inline std::string s_reference_target;

class HashApiTest : public ::testing::Test
{
public:
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
#ifdef BACKEND_BUILD_DIR
    setenv("ICICLE_BACKEND_INSTALL_DIR", BACKEND_BUILD_DIR, 0 /*=replace*/);
#endif
    icicle_load_backend_from_env_or_default();

    const bool is_cuda_registered = is_device_registered("CUDA");
    if (!is_cuda_registered) { ICICLE_LOG_ERROR << "CUDA device not found. Testing CPU vs CPU"; }
    s_main_target = is_cuda_registered ? "CUDA" : "CPU";
    s_reference_target = "CPU";
  }
  static void TearDownTestSuite()
  {
    // make sure to fail in CI if only have one device
    // ICICLE_ASSERT(is_device_registered("CUDA")) << "missing CUDA backend";
  }

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}

  void randomize(uint32_t* arr, uint64_t size)
  {
    // Create a random number generator
    std::random_device rd;                                       // Non-deterministic random number generator
    std::mt19937 gen(rd());                                      // Mersenne Twister engine seeded with rd()
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX); // Range of random numbers

    // Fill the array with random values
    for (uint64_t i = 0; i < size; ++i) {
      arr[i] = dist(gen);
    }
  }
};

TEST_F(HashApiTest, Keccak256)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  // Create unique pointers for input and output arrays
  auto input = std::make_unique<uint32_t[]>(nof_input_limbs);
  auto output = std::make_unique<uint32_t[]>(nof_input_limbs);
  // Randomize the input array
  randomize(input.get(), nof_input_limbs);

  auto config = default_hash_config();
  // Create Keccak-256 hash object
  auto keccak256 = create_keccak_256_hash(nof_input_limbs);
  // Run single hash operation
  ICICLE_CHECK(keccak256.hash_single(input.get(), output.get(), config));
  // TODO: Verify output (e.g., check CPU against CUDA)
}

TEST_F(HashApiTest, Keccak512)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  // Create unique pointers for input and output arrays
  auto input = std::make_unique<uint32_t[]>(nof_input_limbs);
  auto output = std::make_unique<uint32_t[]>(nof_input_limbs);
  // Randomize the input array
  randomize(input.get(), nof_input_limbs);

  auto config = default_hash_config();
  // Create Keccak-512 hash object
  auto keccak512 = create_keccak_512_hash(nof_input_limbs);
  // Run single hash operation
  ICICLE_CHECK(keccak512.hash_single(input.get(), output.get(), config));
  // TODO: Verify output (e.g., check CPU against CUDA)
}

TEST_F(HashApiTest, sha3_256)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  // Create unique pointers for input and output arrays
  auto input = std::make_unique<uint32_t[]>(nof_input_limbs);
  auto output = std::make_unique<uint32_t[]>(nof_input_limbs);
  // Randomize the input array
  randomize(input.get(), nof_input_limbs);

  auto config = default_hash_config();
  // Create sha3-256 hash object
  auto sha3_256 = create_keccak_256_hash(nof_input_limbs);
  // Run single hash operation
  ICICLE_CHECK(sha3_256.hash_single(input.get(), output.get(), config));
  // TODO: Verify output (e.g., check CPU against CUDA)
}

TEST_F(HashApiTest, sha3_512)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  // Create unique pointers for input and output arrays
  auto input = std::make_unique<uint32_t[]>(nof_input_limbs);
  auto output = std::make_unique<uint32_t[]>(nof_input_limbs);
  // Randomize the input array
  randomize(input.get(), nof_input_limbs);

  auto config = default_hash_config();
  // Create sha3-512 hash object
  auto sha3_512 = create_keccak_512_hash(nof_input_limbs);
  // Run single hash operation
  ICICLE_CHECK(sha3_512.hash_single(input.get(), output.get(), config));
  // TODO: Verify output (e.g., check CPU against CUDA)
}