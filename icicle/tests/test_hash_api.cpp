#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <functional>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"
#include "icicle/hash/merkle_tree.h"

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
    std::cout << "Main Target:" << s_main_target << std::endl;
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

eIcicleError run_hash_test(
  const std::function<Hash(uint64_t)>& create_hash_function, const uint64_t nof_input_limbs)
{
  HashConfig config;
  // Create unique pointers for input and output arrays
  auto input = std::make_unique<uint32_t[]>(nof_input_limbs);
  // Randomize the input array
  randomize(input.get(), nof_input_limbs);
  const uint32_t* test_string = input.get();

  // Initialize devices
  Device main_device = {s_main_target, 0};
  Device reference_device = {s_reference_target, 0};

  // Set device to main target and create the main target hash
  icicle_set_device(main_device);
  Hash keccak_cpu_hash = create_hash_function(nof_input_limbs);
  uint8_t cpu_hash[keccak_cpu_hash.total_output_limbs() * sizeof(limb_t)];
  ICICLE_CHECK(keccak_cpu_hash.hash_single(reinterpret_cast<const limb_t*>(test_string), reinterpret_cast<limb_t*>(cpu_hash), config));

  // Set device to reference target and create the reference target hash
  icicle_set_device(reference_device);
  Hash keccak_cuda_hash = create_hash_function(nof_input_limbs);
  uint8_t cuda_hash[keccak_cuda_hash.total_output_limbs() * sizeof(limb_t)];
  ICICLE_CHECK(keccak_cuda_hash.hash_single(reinterpret_cast<const limb_t*>(test_string), reinterpret_cast<limb_t*>(cuda_hash), config));

  // Convert the hashes to strings for comparison
  char computed_cpu_hash_str[keccak_cpu_hash.total_output_limbs() * sizeof(limb_t) * 2 + 1];
  char computed_cuda_hash_str[keccak_cuda_hash.total_output_limbs() * sizeof(limb_t) * 2 + 1];

  for (size_t i = 0; i < keccak_cpu_hash.total_output_limbs() * sizeof(limb_t); i++) {
    sprintf(&computed_cpu_hash_str[i * 2], "%02x", cpu_hash[i]);
    sprintf(&computed_cuda_hash_str[i * 2], "%02x", cuda_hash[i]);
  }
  computed_cpu_hash_str[keccak_cpu_hash.total_output_limbs() * sizeof(limb_t) * 2] = '\0';
  computed_cuda_hash_str[keccak_cuda_hash.total_output_limbs() * sizeof(limb_t) * 2] = '\0';

  // Print the computed hashes
  // std::cout << "Computed CPU hash:  " << computed_cpu_hash_str << std::endl;
  // std::cout << "Computed CUDA hash: " << computed_cuda_hash_str << std::endl;
  icicle_set_device(reference_device);

  // Check if hashes match
  if (strcmp(computed_cpu_hash_str, computed_cuda_hash_str) != 0) {
    std::cerr << "CPU and CUDA Hashes do not match!" << std::endl;
    return eIcicleError::UNKNOWN_ERROR;
  }

  return eIcicleError::SUCCESS;
}


};



TEST_F(HashApiTest, Keccak256)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  ICICLE_CHECK(run_hash_test(create_keccak_256_hash, nof_input_limbs));

}

TEST_F(HashApiTest, Keccak512)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  ICICLE_CHECK(run_hash_test(create_keccak_512_hash, nof_input_limbs));
}

TEST_F(HashApiTest, sha3_256)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  ICICLE_CHECK(run_hash_test(create_sha3_256_hash, nof_input_limbs));

}

TEST_F(HashApiTest, sha3_512)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  ICICLE_CHECK(run_hash_test(create_sha3_512_hash, nof_input_limbs));

}

TEST_F(HashApiTest, blake2s)
{
  const uint64_t nof_input_limbs = 16; // Number of input limbs
  ICICLE_CHECK(run_hash_test(create_blake2s_hash, nof_input_limbs));

}

/****************************** Merkle **********************************/
class HashSumBackend : public HashBackend
{
public:
  HashSumBackend(uint64_t total_input_limbs, uint64_t total_output_limbs, uint64_t total_secondary_input_limbs = 0)
      : HashBackend(total_input_limbs, total_output_limbs, total_secondary_input_limbs)
  {
  }

  virtual eIcicleError hash_single(
    const limb_t* input_limbs,
    limb_t* output_limbs,
    const HashConfig& config,
    const limb_t* secondary_input_limbs = nullptr) const
  {
    output_limbs[0] = 0;
    for (int i = 0; i < m_total_input_limbs; ++i) {
      output_limbs[0] += input_limbs[i];
    }
    for (int i = 1; i < m_total_output_limbs; ++i) {
      output_limbs[i] += output_limbs[0];
    }
    return eIcicleError::SUCCESS;
  }

  virtual eIcicleError hash_many(
    const limb_t* input_limbs,
    limb_t* output_limbs,
    int nof_hashes,
    const HashConfig& config,
    const limb_t* secondary_input_limbs = nullptr) const
  {
    for (int i = 0; i < nof_hashes; ++i) {
      hash_single(input_limbs, output_limbs, config, secondary_input_limbs);
      input_limbs += m_total_input_limbs;
      output_limbs += m_total_output_limbs;
      secondary_input_limbs += m_total_secondary_input_limbs;
    }
    return eIcicleError::SUCCESS;
  }
};

Hash create_hash_sum(uint64_t total_input_limbs, uint64_t total_output_limbs, uint64_t total_secondary_input_limbs = 0)
{
  auto backend = std::make_shared<HashSumBackend>(total_input_limbs, total_output_limbs, total_secondary_input_limbs);
  return Hash(backend);
}

TEST_F(HashApiTest, MerkleTree)
{
  // const uint64_t nof_leaves_limbs = 100; // Number of input limbs
  // auto leaves = std::make_unique<uint32_t[]>(nof_leaves_limbs);
  // // Randomize the input array
  // randomize(leaves.get(), nof_leaves_limbs);

  auto config = default_merkle_tree_config();
  auto layer0_hash = create_hash_sum(5, 2); // input 5 limbs, output 2 limbs
  auto layer1_hash = create_hash_sum(6, 3); // input 6 limbs, output 3 limbs
  auto layer2_hash = create_hash_sum(6, 2); // input 6 limbs, output 2 limbs
  auto merkle_tree =
    create_merkle_tree({layer0_hash, layer1_hash, layer2_hash}, 1 /*limbs per leaf*/, 2 /*min level to store*/);

  limb_t leaves[100];
  for (int i = 0; i < 100; ++i) {
    leaves[i] = i;
  }
  // std::cout << “build start” << std::endl;
  merkle_tree.build(leaves, config);
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