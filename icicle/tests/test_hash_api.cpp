#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/merkle/merkle_tree.h"

#include <string>
#include <sstream>
#include <iomanip>

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
static inline std::vector<std::string> s_registered_devices;

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
    s_registered_devices = get_registered_devices_list();
    ASSERT_GE(s_registered_devices.size(), 1);
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
    for (int i = 0; i < (size * sizeof(T) / sizeof(uint32_t)); ++i) {
      u32_arr[i] = dist(gen);
    }
  }

  std::string voidPtrToHexString(const std::byte* byteData, size_t size)
  {
    std::ostringstream hexStream;
    for (size_t i = 0; i < size; ++i) {
      // Use fully qualified names for std::hex, std::setw, and std::setfill
      hexStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byteData[i]);
    }

    return hexStream.str();
  }
};

TEST_F(HashApiTest, Keccak512)
{
  auto config = default_hash_config();

  const std::string input = "HelloWorld! FromKeccak512";
  const std::string expected_output = "b493094fc34b23cc868b170f68b767fcd5844f51640fdce7946958aba24336007637325d567ae456"
                                      "d4c981f144031a398f37122eb476fe75a67ab85974098e9a";
  const uint64_t output_size = 64;
  auto output = std::make_unique<std::byte[]>(output_size);

  // Create Keccak-512 hash object
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Keccak512 test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    auto keccak512 = Keccak512::create();
    ICICLE_CHECK(keccak512.hash(input.data(), input.size() / config.batch, config, output.get()));
    // Convert the output do a hex string and compare to expected output string
    std::string output_as_str = voidPtrToHexString(output.get(), output_size);
    ASSERT_EQ(output_as_str, expected_output);
  }
}

TEST_F(HashApiTest, Keccak256Batch)
{
  auto config = default_hash_config();
  config.batch = 2;

  const std::string input = "0123456789abcdef"; // this is a batch of "01234567" and "89abcdef"
  const std::string expected_output_0 = "d529b8ccadec912a5c302a7a9ef53e70c144eea6043dcea534fdbbb2d042fc31";
  const std::string expected_output_1 = "58ed472a16d883f4dec9fc40438a59b017de9a7dbaa0bbc2cc9170e94eed2337";
  const uint64_t output_size = 32;
  auto output = std::make_unique<std::byte[]>(output_size * config.batch);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Keccak256 test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // Create Keccak-256 hash object and hash
    auto keccak256 = Keccak256::create();
    ICICLE_CHECK(keccak256.hash(input.data(), input.size() / config.batch, config, output.get()));
    // Convert the output do a hex string and compare to expected output string
    std::string output_as_str = voidPtrToHexString(output.get(), output_size);
    ASSERT_EQ(output_as_str, expected_output_0);
    output_as_str = voidPtrToHexString(output.get() + output_size, output_size);
    ASSERT_EQ(output_as_str, expected_output_1);
  }
}

TEST_F(HashApiTest, KeccakLarge)
{
  auto config = default_hash_config();
  config.batch = 1 << 8;
  const unsigned chunk_size = 1 << 13; // 8KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(s_reference_target));
  auto keccakCPU = Keccak256::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(keccakCPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU Keccak large time", true);

  ICICLE_CHECK(icicle_set_device(s_main_target));
  auto keccakCUDA = Keccak256::create();

  // test with host memory
  START_TIMER(cuda_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(keccakCUDA.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(cuda_timer, "CUDA Keccak large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(cuda_timer_device_mem);
  ICICLE_CHECK(keccakCUDA.hash(d_input, chunk_size, config, d_output));
  END_TIMER(cuda_timer_device_mem, "CUDA Keccak large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));
}

TEST_F(HashApiTest, sha3)
{
  auto config = default_hash_config();

  const std::string input = "I am SHA3";
  const uint64_t output_size = 64;
  auto output = std::make_unique<std::byte[]>(output_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "SHA3 test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    // sha3-256
    auto sha3_256 = Sha3_256::create();
    ICICLE_CHECK(sha3_256.hash(input.data(), input.size() / config.batch, config, output.get()));
    const std::string expected_output_sha_256 = "b45ee6bc2e599daf8ffd1fd952c32f58e6a7046300331b2321b927327a9affcf";
    std::string output_as_str = voidPtrToHexString(output.get(), 32);
    ASSERT_EQ(output_as_str, expected_output_sha_256);
    // sha3-512
    auto sha3_512 = Sha3_512::create();
    ICICLE_CHECK(sha3_512.hash(input.data(), input.size() / config.batch, config, output.get()));
    const std::string expected_output_sha_512 =
      "50b0cf05a243907301a10a1c14b4750a8fdbd1f8ef818624dff2f4e83901c9f8e8de84a2"
      "410d45c968b9307dfd9a4da58768e0d1f5594511b31b7274cfc04280";
    output_as_str = voidPtrToHexString(output.get(), 64);
    ASSERT_EQ(output_as_str, expected_output_sha_512);
  }
}

/****************************** Merkle **********************************/
class HashSumBackend : public HashBackend
{
public:
  HashSumBackend(uint64_t input_chunk_size, uint64_t output_size)
      : HashBackend("HashSumTest", output_size, input_chunk_size)
  {
  }

  eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
  {
    const auto chunk_size = get_single_chunk_size(size);
    const auto otput_digest_size = output_size();
    for (int i = 0; i < config.batch; ++i) {
      hash_single(input, size, config, output);
      input += chunk_size;
      output += otput_digest_size;
    }
    return eIcicleError::SUCCESS;
  }

  void hash_single(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const
  {
    const uint32_t* input_u32 = (const uint32_t*)input;
    uint32_t* output_u32 = (uint32_t*)output;

    output_u32[0] = 0;
    for (int i = 0; i < (size >> 2); ++i) {
      output_u32[0] += input_u32[i];
    }
    for (int i = 1; i < (output_size() >> 2); ++i) {
      output_u32[i] += output_u32[0];
    }
  }

  static Hash create(uint64_t input_chunk_size, uint64_t output_size)
  {
    auto backend = std::make_shared<HashSumBackend>(input_chunk_size, output_size);
    return Hash(backend);
  }
};

TEST_F(HashApiTest, MerkleTree)
{
  ICICLE_CHECK(icicle_set_device(s_reference_target)); // TODO CUDA too
  // define input
  constexpr int nof_leaves = 100;
  uint32_t leaves[nof_leaves];
  const size_t input_size = sizeof(leaves);
  for (int i = 0; i < nof_leaves; ++i) {
    leaves[i] = i;
  }

  // define the merkle tree
  auto config = default_merkle_tree_config();
  auto layer0_hash = HashSumBackend::create(20, 8); // input 20 bytes, output 8 bytes input   400B ->  160B
  auto layer1_hash = HashSumBackend::create(16, 2); // input 16 bytes, output 2 bytes         160B ->  20B
  auto layer2_hash = HashSumBackend::create(20, 8); // input 20 bytes, output 8 bytes         20B  ->  8B    output
  auto leaf_element_size = 4;
  auto merkle_tree =
    MerkleTree::create({layer0_hash, layer1_hash, layer2_hash}, leaf_element_size, 2 /*min level to store*/);

  // build tree
  ICICLE_CHECK(merkle_tree.build(leaves, input_size, config));

  // get root and merkle-path to an element
  uint64_t leaf_idx = 5;
  auto [root, root_size] = merkle_tree.get_merkle_root();
  MerkleProof merkle_proof{};
  ICICLE_CHECK(merkle_tree.get_merkle_proof(leaves, leaf_idx, config, merkle_proof));

  bool verification_valid = false;
  ICICLE_CHECK(merkle_tree.verify(merkle_proof, verification_valid));
  ASSERT_TRUE(verification_valid);
}

#ifdef POSEIDON

#include "icicle/fields/field_config.h"
using namespace field_config;

#include "icicle/hash/poseidon.h"

TEST_F(HashApiTest, poseidon12)
{
  const uint64_t arity = 12; // Number of input elements

  // Create unique pointers for input and output arrays
  auto input = std::make_unique<scalar_t[]>(arity);
  scalar_t output = scalar_t::from(0);
  // Randomize the input array
  scalar_t::rand_host_many(input.get(), arity);

  // init poseidon constants on current device
  ICICLE_CHECK(Poseidon::init_default_constants<scalar_t>());

  // Create Poseidon hash object
  auto poseidon = Poseidon::create<scalar_t>(arity);

  // Run single hash operation
  auto config = default_hash_config();
  ICICLE_CHECK(poseidon.hash(input.get(), arity, config, &output));
  // TODO: Verify output (e.g., check CPU against CUDA)
}
#endif // POSEIDON