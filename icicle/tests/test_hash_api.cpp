#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/fields/field.h"

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
static int ITERS = 1;
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
    ICICLE_ASSERT(is_device_registered("CUDA")) << "missing CUDA backend";
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

TEST_F(HashApiTest, Blake2s)
{
  auto config = default_hash_config();

  const std::string input = "Hello world I am blake2s";
  const std::string expected_output = "291c4b3648438cc57d1e965ee52e5572e8dc4938bc960e22d6ebe3a280aea759";

  const uint64_t output_size = 32;
  auto output = std::make_unique<std::byte[]>(output_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Blake2s test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    auto blake2s = Blake2s::create();
    ICICLE_CHECK(blake2s.hash(input.data(), input.size() / config.batch, config, output.get()));
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

TEST_F(HashApiTest, Blake2sLarge)
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
  auto blake2sCPU = Blake2s::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(blake2sCPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU blake2s large time", true);

  ICICLE_CHECK(icicle_set_device(s_main_target));
  auto blake2sCUDA = Blake2s::create();

  // test with host memory
  START_TIMER(cuda_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(blake2sCUDA.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(cuda_timer, "CUDA blake2s large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(cuda_timer_device_mem);
  ICICLE_CHECK(blake2sCUDA.hash(d_input, chunk_size, config, d_output));
  END_TIMER(cuda_timer_device_mem, "CUDA blake2s large time (on device memory)", true);
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
      : HashBackend("HashSum", output_size, input_chunk_size)
  {
  }

  eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
  {
    const auto chunk_size = get_single_chunk_size(size);
    const auto output_digest_size = output_size();
    for (int i = 0; i < config.batch; ++i) {
      hash_single(input, size, config, output);
      input += chunk_size;
      output += output_digest_size;
    }
    return eIcicleError::SUCCESS;
  }

  void hash_single(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const
  {
    const uint32_t* input_u32 = (const uint32_t*)input;
    uint32_t* output_u32 = (uint32_t*)output;

    output_u32[0] = 0;
    int t_size = sizeof(uint32_t);
    for (int i = 0; i < (size / t_size); ++i) {
      output_u32[0] += input_u32[i];
    }
    for (int i = 1; i < (output_size() / t_size); ++i) {
      output_u32[i] = output_u32[0];
    }
  }

  static Hash create(uint64_t input_chunk_size, uint64_t output_size)
  {
    auto backend = std::make_shared<HashSumBackend>(input_chunk_size, output_size);
    return Hash(backend);
  }
};

void assert_valid_tree(
  const MerkleTree& tree,
  int input_size,
  const std::byte* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config)
{
  ICICLE_ASSERT(!hashes.empty());
  int output_size = input_size * hashes[0].output_size() / hashes[0].default_input_chunk_size();
  auto layer_in =
    std::make_unique<std::byte[]>(input_size); // Going layer by layer - having the input layer as the largest
  auto layer_out =
    std::make_unique<std::byte[]>(output_size); // ensures these are the maximum sizes required for the arrays
  // NOTE there is an assumption here that output number is less or equal to input number for all layers

  memcpy(layer_in.get(), inputs, input_size);

  int side_inputs_offset = 0;
  for (auto& layer_hash : hashes) {
    output_size = input_size * layer_hash.output_size() / layer_hash.default_input_chunk_size();
    const int nof_hashes = input_size / layer_hash.default_input_chunk_size();

    auto config = default_hash_config();
    config.batch = nof_hashes;
    layer_hash.hash(layer_in.get(), layer_hash.default_input_chunk_size(), config, layer_out.get());

    // copy output outputs to inputs before moving to the next layer
    memcpy(layer_in.get(), layer_out.get(), output_size);
    input_size = output_size;
  }

  // Compare computed root with the tree's root
  auto [root, root_size] = tree.get_merkle_root();

  for (int i = 0; i < root_size; i++) {
    ASSERT_EQ(root[i], layer_out[i]) << "Mismatch in root[" << i << "]";
  }
}

template <typename T>
void assert_valid_tree(
  const MerkleTree& tree,
  int nof_inputs,
  const T* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config)
{
  return assert_valid_tree(tree, nof_inputs * sizeof(T), reinterpret_cast<const std::byte*>(inputs), hashes, config);
}

TEST_F(HashApiTest, MerkleTreeBasic)
{
  const int leaf_size = sizeof(uint32_t);
  const int nof_leaves = 50;
  uint32_t leaves[nof_leaves];
  randomize(leaves, nof_leaves);

  uint32_t leaves_alternative[nof_leaves];
  randomize(leaves_alternative, nof_leaves);

  ICICLE_CHECK(icicle_set_device(s_reference_target));

  // define the merkle tree
  auto config = default_merkle_tree_config();
  auto layer0_hash = HashSumBackend::create(5 * leaf_size, 2 * leaf_size); // in 5 leaves, out 2 leaves  200B ->  80B
  auto layer1_hash = HashSumBackend::create(4 * leaf_size, leaf_size);     // in 4 leaves, out 1 leaf    80B  ->  20B
  auto layer2_hash = HashSumBackend::create(leaf_size, leaf_size);         // in 1 leaf, out 1 leaf      20B  ->  20B
  auto layer3_hash = HashSumBackend::create(5 * leaf_size, leaf_size); // in 5 leaves, out 1 leaf    20B  ->  4B output

  std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};

  int output_store_min_layer;
  randomize(&output_store_min_layer, 1);
  output_store_min_layer = output_store_min_layer & 3; // Ensure index is in a valid 0-3 range
  ICICLE_LOG_DEBUG << "Min store layer:\t" << output_store_min_layer;

  auto prover_tree = MerkleTree::create(hashes, sizeof(uint32_t), output_store_min_layer);
  auto verifier_tree = MerkleTree::create(hashes, sizeof(uint32_t), output_store_min_layer);

  // build tree
  ICICLE_CHECK(prover_tree.build(leaves, nof_leaves, config));
  assert_valid_tree<uint32_t>(prover_tree, nof_leaves, leaves, hashes, config);

  // get root and merkle-path for a leaf
  const int nof_leaves_to_test = 5;
  uint64_t leaf_indices[nof_leaves_to_test];
  randomize(leaf_indices, nof_leaves_to_test);

  for (int i = 0; i < nof_leaves_to_test; i++) {
    int leaf_idx = leaf_indices[i] % nof_leaves;

    auto [root, root_size] = prover_tree.get_merkle_root();
    MerkleProof merkle_proof{};
    ICICLE_CHECK(prover_tree.get_merkle_proof(leaves, nof_leaves, leaf_idx, false, config, merkle_proof));

    // Test valid proof
    bool verification_valid = false;
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid);

    // Test invalid proof (By modifying random data in the leaves)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(leaves_alternative, nof_leaves, leaf_idx, false, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid);

    // Same for pruned proof
    verification_valid = false;
    ICICLE_CHECK(prover_tree.get_merkle_proof(leaves, nof_leaves, leaf_idx, true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid);

    // Test invalid proof (By adding random data to the proof)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(leaves_alternative, nof_leaves, leaf_idx, true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid);
  }
}

TEST_F(HashApiTest, MerkleTreeMixMediumSize)
{
  const uint32_t leaf_size = sizeof(uint32_t);
  const uint32_t total_input_size = (1 << 20);
  const uint32_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<uint32_t[]>(nof_leaves);
  randomize(leaves.get(), nof_leaves);

  auto leaves_alternative = std::make_unique<uint32_t[]>(nof_leaves);
  randomize(leaves_alternative.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeMixMediumSize on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // define the merkle tree
    auto config = default_merkle_tree_config();

    auto layer0_hash = Keccak256::create(1 << 10); // hash every 1KB to 32B -> layer outputs 32KB (for 1MB input)
    auto layer1_hash = Blake2s::create(32 * 2);    // arity-2: 32KB -> 16KB
    auto layer2_hash = Sha3_512::create(32 * 4);   // arity-4: 16KB -> 8KB (note output is 64B per hash in this layer)
    auto layer3_hash = Blake2s::create(64 * 4);    // arity-4: 8KB -> 1KB
    auto layer4_hash = Keccak512::create(32 * 32); // arity-32: 1KB -> 64B

    const std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash, layer4_hash};

    const int output_store_min_layer = rand() % hashes.size();
    ICICLE_LOG_DEBUG << "Min store layer:\t" << output_store_min_layer;

    auto prover_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);
    auto verifier_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);

    // assert that incorrect size fails
    ASSERT_NE(prover_tree.build(leaves.get(), nof_leaves - 1, config), eIcicleError::SUCCESS);
    ASSERT_NE(prover_tree.build(leaves.get(), nof_leaves + 1, config), eIcicleError::SUCCESS);
    // build tree
    START_TIMER(MerkleTree_build)
    ICICLE_CHECK(prover_tree.build(leaves.get(), nof_leaves, config));
    END_TIMER(MerkleTree_build, "Merkle Tree large", true)
    assert_valid_tree<uint32_t>(prover_tree, nof_leaves, leaves.get(), hashes, config);

    // get root and merkle-path to an element
    for (int test_leaf_idx = 0; test_leaf_idx < 5; test_leaf_idx++) {
      const int leaf_idx = rand() % nof_leaves;

      auto [root, root_size] = prover_tree.get_merkle_root();
      MerkleProof merkle_proof{};
      ICICLE_CHECK(prover_tree.get_merkle_proof(leaves.get(), nof_leaves, leaf_idx, false, config, merkle_proof));

      // Test valid proof
      bool verification_valid = false;
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);

      // Test invalid proof (By modifying random data in the leaves)
      verification_valid = true;
      ICICLE_CHECK(prover_tree.get_merkle_proof(
        leaves_alternative.get(), nof_leaves, leaf_idx, false /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_FALSE(verification_valid);

      // Same for pruned proof
      verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(leaves.get(), nof_leaves, leaf_idx, true /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);

      // Test invalid proof (By adding random data to the proof)
      verification_valid = true;
      ICICLE_CHECK(prover_tree.get_merkle_proof(
        leaves_alternative.get(), nof_leaves, leaf_idx, true /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_FALSE(verification_valid);
    }
  }
}

TEST_F(HashApiTest, MerkleTreeDevicePartialTree)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 8) * leaf_size;
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceCaps on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
    auto layer_hash = Keccak256::create(32 * 2);
    // Calculate the tree height (log2 of the number of leaves for a binary tree)
    const int tree_height = static_cast<int>(std::log2(nof_leaves));
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(tree_height, layer_hash);

    auto config = default_merkle_tree_config();
    auto full_tree = MerkleTree::create(hashes, leaf_size);
    auto prover_tree = MerkleTree::create(hashes, leaf_size, 4);
    auto verifier_tree = MerkleTree::create(hashes, leaf_size, 4);

    // build tree
    ICICLE_CHECK(prover_tree.build(leaves.get(), total_input_size, config));
    ICICLE_CHECK(full_tree.build(leaves.get(), total_input_size, config));

    auto full_root = full_tree.get_merkle_root();
    auto partial_root = prover_tree.get_merkle_root();
    for (int i = 0; i < full_root.second; i++) {
      ASSERT_TRUE(full_root.first[i] == partial_root.first[i]);
    }

    // proof leaves and verify
    for (int test_leaf_idx = 0; test_leaf_idx < 5; test_leaf_idx++) {
      const int leaf_idx = rand() % nof_leaves;

      auto [root, root_size] = prover_tree.get_merkle_root();

      // test non-pruned path
      MerkleProof merkle_proof{};
      bool verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(leaves.get(), nof_leaves, leaf_idx, false /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);

      // test pruned path
      verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(leaves.get(), nof_leaves, leaf_idx, true /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);
    }
  }
}

TEST_F(HashApiTest, MerkleTreeLeavesOnDeviceTreeOnHost)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 3) * leaf_size;
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceBig on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
    auto layer_hash = Keccak256::create(32 * 2);
    // Calculate the tree height (log2 of the number of leaves for a binary tree)
    const int tree_height = static_cast<int>(std::log2(nof_leaves));
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(tree_height, layer_hash);

    // copy leaves to device
    std::byte* device_leaves = nullptr;
    ICICLE_CHECK(icicle_malloc((void**)&device_leaves, total_input_size));
    ICICLE_CHECK(icicle_copy(device_leaves, leaves.get(), total_input_size));

    auto config = default_merkle_tree_config();
    config.is_tree_on_device = false;
    config.is_leaves_on_device = true;
    auto prover_tree = MerkleTree::create(hashes, leaf_size);
    auto verifier_tree = MerkleTree::create(hashes, leaf_size);

    // build tree
    START_TIMER(MerkleTree_build)
    ICICLE_CHECK(prover_tree.build(device_leaves, total_input_size, config));
    END_TIMER(MerkleTree_build, "Merkle Tree GPU", true)

    // proof leaves and verify
    for (int test_leaf_idx = 0; test_leaf_idx < 5; test_leaf_idx++) {
      const int leaf_idx = rand() % nof_leaves;

      auto [root, root_size] = prover_tree.get_merkle_root();

      // test non-pruned path
      MerkleProof merkle_proof{};
      bool verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(device_leaves, nof_leaves, leaf_idx, false /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);

      // test pruned path
      verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(device_leaves, nof_leaves, leaf_idx, true /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);
    }
    ICICLE_CHECK(icicle_free(device_leaves));
  }
}

TEST_F(HashApiTest, MerkleTreeLarge)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 28);
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceBig on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
    auto layer_hash = Keccak256::create(32 * 2);
    // Calculate the tree height (log2 of the number of leaves for a binary tree)
    const int tree_height = static_cast<int>(std::log2(nof_leaves));
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(tree_height, layer_hash);

    // copy leaves to device
    std::byte* device_leaves = nullptr;
    ICICLE_CHECK(icicle_malloc((void**)&device_leaves, total_input_size));
    ICICLE_CHECK(icicle_copy(device_leaves, leaves.get(), total_input_size));

    auto config = default_merkle_tree_config();
    config.is_leaves_on_device = true;
    auto prover_tree = MerkleTree::create(hashes, leaf_size);
    auto verifier_tree = MerkleTree::create(hashes, leaf_size);

    // build tree
    START_TIMER(MerkleTree_build)
    ICICLE_CHECK(prover_tree.build(device_leaves, total_input_size, config));
    END_TIMER(MerkleTree_build, "Merkle Tree large", true)

    // proof leaves and verify
    for (int test_leaf_idx = 0; test_leaf_idx < 5; test_leaf_idx++) {
      const int leaf_idx = rand() % nof_leaves;

      auto [root, root_size] = prover_tree.get_merkle_root();

      // test non-pruned path
      MerkleProof merkle_proof{};
      bool verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(device_leaves, nof_leaves, leaf_idx, false /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);

      // test pruned path
      verification_valid = false;
      ICICLE_CHECK(
        prover_tree.get_merkle_proof(device_leaves, nof_leaves, leaf_idx, true /*=pruned*/, config, merkle_proof));
      ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
      ASSERT_TRUE(verification_valid);
    }
    ICICLE_CHECK(icicle_free(device_leaves));
  }
}

#ifdef POSEIDON2
// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001

#include "icicle/fields/field_config.h"
#include "icicle/hash/poseidon2_constants/constants/bn254_poseidon2.h"

using namespace field_config;
using namespace poseidon2_constants_bn254;

#include "icicle/hash/poseidon2.h"

// 0x303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570

TEST_F(HashApiTest, poseidon2_3_single_hash_cpu_only)
{
  const unsigned t = 3;
  // const unsigned t = 12;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t);
  scalar_t::rand_host_many(input.get(), t);
    // DEBUG
  input[0] = scalar_t::from(0);
  input[1] = scalar_t::from(1);
  input[2] = scalar_t::from(2);
  // DEBUG
  config.batch = 1;

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_3_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_3_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  // auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  scalar_t expected_res = scalar_t::hex_str2scalar("0x303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570");
  // run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  ASSERT_EQ(expected_res, *(output_cpu.get()));

  // ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

// Test check single hash without domain tag.
TEST_F(HashApiTest, poseidon2_3_single_hash_without_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t);
  scalar_t::rand_host_many(input.get(), t);
    // DEBUG
  input[0] = scalar_t::from(0);
  input[1] = scalar_t::from(1);
  input[2] = scalar_t::from(2);
  // DEBUG
  config.batch = 1;

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_3_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_3_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon2_3_single_hash_with_dt)    // Danny - TODO. To add domain_tag test
{
  // const unsigned t = 3;
  // auto config = default_hash_config();

  // auto input = std::make_unique<scalar_t[]>(t);
  // scalar_t::rand_host_many(input.get(), t);
  //   // DEBUG
  // input[0] = scalar_t::from(0);
  // input[1] = scalar_t::from(1);
  // input[2] = scalar_t::from(2);
  // // DEBUG
  // config.batch = 1;

  // auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
  //   Device dev = {dev_type, 0};
  //   icicle_set_device(dev);

  //   std::ostringstream oss;
  //   oss << dev_type << " " << msg;

  //   auto poseidon2 = Poseidon2::create<scalar_t>(t);

  //   START_TIMER(POSEIDON2_3_sync)
  //   for (int i = 0; i < iters; ++i) {
  //     ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
  //   }
  //   END_TIMER(POSEIDON2_3_sync, oss.str().c_str(), measure);
  // };

  // auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  // auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  // run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  // run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  // ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon2_3_batch)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  const scalar_t domain_tag = scalar_t::rand_host();

  config.batch = 1 << 10;
  config.batch = 4;     // DEBUG. To delete.
  // auto input = std::make_unique<scalar_t[]>((t - 1) * config.batch);     // DEBUG
  // scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);   // DEBUG
  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    std::cout << "iters = " << iters << std::endl;
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      // std::cout << "poseidon2_3_batch: t - 1 = " << std::dec << t - 1 << std::endl;    // DEBUG
      // ICICLE_CHECK(poseidon2.hash(input.get(), t - 1, config, out));   // DEBUG
      std::cout << "poseidon2_3_batch: t = " << std::dec << t << std::endl;
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon2_3_2_two_level_tree)   // Danny- TODO. To add test with two levels tree with diff t.
{
  // const unsigned t = 3;
  // auto config = default_hash_config();
  // const scalar_t domain_tag = scalar_t::rand_host();

  // config.batch = 1 << 10;
  // config.batch = 4;     // DEBUG. To delete.
  // // auto input = std::make_unique<scalar_t[]>((t - 1) * config.batch);     // DEBUG
  // // scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);   // DEBUG
  // auto input = std::make_unique<scalar_t[]>(t * config.batch);
  // scalar_t::rand_host_many(input.get(), t * config.batch);

  // auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
  //   std::cout << "iters = " << iters << std::endl;
  //   Device dev = {dev_type, 0};
  //   icicle_set_device(dev);

  //   std::ostringstream oss;
  //   oss << dev_type << " " << msg;

  //   auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);

  //   START_TIMER(POSEIDON2_sync)
  //   for (int i = 0; i < iters; ++i) {
  //     // std::cout << "poseidon2_3_batch: t - 1 = " << std::dec << t - 1 << std::endl;    // DEBUG
  //     // ICICLE_CHECK(poseidon2.hash(input.get(), t - 1, config, out));   // DEBUG
  //     std::cout << "poseidon2_3_batch: t = " << std::dec << t << std::endl;
  //     ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
  //   }
  //   END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  // };

  // auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  // auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  // run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  // run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  // ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}
#endif    // POSEIDON2

#ifdef POSEIDON
// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001

  #include "icicle/fields/field_config.h"
  #include "poseidon/constants/bn254_poseidon.h"

using namespace field_config;
using namespace poseidon_constants_bn254;

  #include "icicle/hash/poseidon.h"

TEST_F(HashApiTest, poseidon_12_single_hash)
{
  const unsigned t = 12;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t);
  scalar_t::rand_host_many(input.get(), t);
  config.batch = 1;

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

// TEST_F(HashApiTest, poseidon_3_single_hash_domain_tag)
// {
//   const unsigned  t                   = 2;
//   const unsigned  default_input_size      = 2;
//   const bool      use_domain_tag           = true;
//   scalar_t        domain_tag_value        = scalar_t::from(7);
//   const bool      use_all_zeroes_padding  = true;
//   auto            config                  = default_hash_config();

//   auto input = std::make_unique<scalar_t[]>(t);
//   scalar_t::rand_host_many(input.get(), t);

//   config.batch = 1;

//   auto run =
//     [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
//       Device dev = {dev_type, 0};
//       icicle_set_device(dev);

//       std::ostringstream oss;
//       oss << dev_type << " " << msg;

//       auto poseidon = Poseidon::create<scalar_t>(t);

//       START_TIMER(POSEIDON_sync)
//       for (int i = 0; i < iters; ++i) {
//         ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
//       }
//       END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
//     };

//   auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
//   auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

//   run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
//   run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

//   ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
// }

TEST_F(HashApiTest, poseidon_3_single_hash)
{
  const unsigned t = 3;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t);
  scalar_t::rand_host_many(input.get(), t);
  config.batch = 1;

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon_3_batch_without_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();

  config.batch = 1 << 10;
  config.batch = 6;
  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    std::cout << "iters = " << iters << std::endl;
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon_3_batch_with_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  const scalar_t domain_tag = scalar_t::rand_host();

  config.batch = 1 << 10;
  config.batch = 4;
  auto input = std::make_unique<scalar_t[]>((t - 1) * config.batch);
  scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    std::cout << "iters = " << iters << std::endl;
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t, &domain_tag);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t - 1, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_cuda = std::make_unique<scalar_t[]>(config.batch);

  run(s_reference_target, output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(s_main_target, output_cuda.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_cuda.get(), config.batch * sizeof(scalar_t)));
}

#endif // POSEIDON