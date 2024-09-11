#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/merkle/merkle_tree.h"

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

  template <typename T>
  void randomize(T* arr, uint64_t size)
  {
    // Create a random number generator
    std::random_device rd;                                       // Non-deterministic random number generator
    std::mt19937 gen(rd());                                      // Mersenne Twister engine seeded with rd()
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX); // Range of random numbers

    // Fill the array with random values
    uint32_t* u32_arr = (uint32_t*)arr;
    for (uint64_t i = 0; i < (size / sizeof(uint32_t)); ++i) {
      u32_arr[i] = dist(gen);
    }
  }
};

TEST_F(HashApiTest, Keccak256)
{
  const uint64_t input_chunk_size = 64; // Number of input bytes
  const uint64_t output_size = 256;

  // Create unique pointers for input and output arrays
  auto input = std::make_unique<std::byte[]>(input_chunk_size);
  auto output = std::make_unique<std::byte[]>(output_size);
  // Randomize the input array
  randomize(input.get(), input_chunk_size);

  auto config = default_hash_config();
  config.input_chunk_size = input_chunk_size;
  // Create Keccak-256 hash object
  auto keccak256 = create_keccak_256_hash();
  // Run single hash operation
  ICICLE_CHECK(keccak256.hash(input.get(), config, output.get()));
  // TODO: Verify output (e.g., check CPU against CUDA)
}

// TODO: add tests for all hashes

/****************************** Merkle **********************************/
class HashSumBackend : public HashBackend
{
public:
  HashSumBackend(uint64_t input_chunk_size, uint64_t output_size) : HashBackend(output_size, input_chunk_size) {}

  eIcicleError hash(const std::byte* input, const HashConfig& config, std::byte* output) const override
  {
    const auto chunk_size = input_chunk_size(config);
    const auto otput_digest_size = output_size();
    for (int i = 0; i < config.batch; ++i) {
      hash_single(input, config, output);
      input += chunk_size;
      output += otput_digest_size;
    }
    return eIcicleError::SUCCESS;
  }

  void hash_single(const std::byte* input, const HashConfig& config, std::byte* output) const
  {
    const auto chunk_size = input_chunk_size(config);
    const uint32_t* input_u32 = (const uint32_t*)input;
    uint32_t* output_u32 = (uint32_t*)output;

    output_u32[0] = 0;
    for (int i = 0; i < (chunk_size >> 2); ++i) {
      output_u32[0] += input_u32[i];
    }
    for (int i = 1; i < (output_size() >> 2); ++i) {
      output_u32[i] += output_u32[0];
    }
  }
};

Hash create_hash_sum(uint64_t input_chunk_size, uint64_t output_size_limbs)
{
  auto backend = std::make_shared<HashSumBackend>(input_chunk_size, output_size_limbs);
  return Hash(backend);
}

TEST_F(HashApiTest, MerkleTree)
{
  // const uint64_t nof_leaves_limbs = 100; // Number of input limbs
  // auto leaves = std::make_unique<uint32_t[]>(nof_leaves_limbs);
  // // Randomize the input array
  // randomize(leaves.get(), nof_leaves_limbs);

  auto config = default_merkle_tree_config();
  auto layer0_hash = create_hash_sum(20, 8);  // input 20 bytes, output 8 bytes
  auto layer1_hash = create_hash_sum(24, 12); // input 24 bytes, output 12 bytes
  auto layer2_hash = create_hash_sum(24, 8);  // input 24 bytes, output 8 bytes
  auto merkle_tree =
    create_merkle_tree({layer0_hash, layer1_hash, layer2_hash}, 1 /*limbs per leaf*/, 2 /*min level to store*/);

  uint32_t leaves[100];
  for (int i = 0; i < 100; ++i) {
    leaves[i] = i;
  }
  // std::cout << “build start” << std::endl;
  merkle_tree.build((std::byte*)leaves, config);
  // merkle_tree.print_tree();
  // const limb_t* root;
  // merkle_tree.get_root(root);
  // std::cout << "Root = " << *root << std::endl;
  // limb_t* path;
  // uint path_size;
  // uint64_t element_idx = 5;
  // merkle_tree.allocate_path(path, path_size);
  // merkle_tree.get_path(leaves, element_idx, path, config);
  // std::cout << “path(“<< element_idx << “):” << std::endl;
  // merkle_tree.print_path(path);
  // std::cout << “verify:” << std::endl;
  // bool verification_valid;
  // merkle_tree.verify(path, element_idx, verification_valid, config);
  // std::cout << “verify = ” << verification_valid<< std::endl;
}