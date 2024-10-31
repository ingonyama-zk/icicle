#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"
#include "icicle/merkle/merkle_tree.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

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
  static void randomize(T* arr, uint64_t size)
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

/**
 * @brief Builds tree in a straight-forward single-threaded manner and compares the result with Icicle's calculation.
 * @param tree - Merkle tree to test.
 * @param input_size - Size of input in bytes.
 * @param inputs - Input as a byte array.
 * @param hashes - Vector of hashes of each layer in the tree above.
 * @return True if the tree's calculations (icicle and test) match.
 */
bool is_valid_tree(
  const MerkleTree& tree,
  int input_size,
  int leaf_size,
  const std::byte* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config)
{
  std::vector<std::byte> input_vec(input_size);
  memcpy(input_vec.data(), inputs, input_size);

  int nof_hashes = 1;
  for (int i = hashes.size() - 2; i >= 0; i--) {
    nof_hashes *= hashes[i + 1].input_default_chunk_size() / hashes[i].output_size();
  }
  int tree_input_size = nof_hashes * hashes[0].input_default_chunk_size();

  // ICICLE_LOG_INFO << "Nof inputs required for tree: " << tree_input_size;
  ICICLE_ASSERT((config.padding_policy != PaddingPolicy::None) || (input_size == tree_input_size))
    << "Leaves size (" << (input_size / leaf_size) << ") is smaller than tree size (" << (tree_input_size / leaf_size)
    << ") while Padding policy is None\n";

  if (tree_input_size > input_size) {
    input_vec.resize(tree_input_size);
    if (config.padding_policy == PaddingPolicy::LastValue) {
      ICICLE_ASSERT(tree_input_size % leaf_size == 0)
        << "Leaf size (" << leaf_size << ") must divide tree size (" << tree_input_size << ")";
      std::vector<std::byte> last_leaf(leaf_size);
      memcpy(last_leaf.data(), inputs + input_size - leaf_size, leaf_size);
      int nof_leaves_in_padding = (tree_input_size - input_size) / leaf_size;
      for (int i = 0; i < nof_leaves_in_padding; i++) {
        memcpy(input_vec.data() + input_size + i * leaf_size, last_leaf.data(), leaf_size);
      }
    }
  }

  // std::cout << "Padded inputs:\n";
  // for (int i = 0; i < input_vec.size(); i++)
  // {
  //   std::cout << std::to_integer<int>(input_vec[i]) << "\t";
  // }
  // std::cout << '\n';

  int max_layer_size_bytes = input_vec.size();
  int input_size_temp = input_vec.size();
  int output_size_temp = 1;

  for (auto& layer_hash : hashes) {
    output_size_temp = input_size_temp * layer_hash.output_size() / layer_hash.input_default_chunk_size();
    if (output_size_temp > max_layer_size_bytes) { max_layer_size_bytes = output_size_temp; }

    input_size_temp = output_size_temp;
  }

  input_size_temp = input_vec.size();
  int output_size = input_size * hashes[0].output_size() / hashes[0].input_default_chunk_size();
  auto layer_in = std::make_unique<std::byte[]>(max_layer_size_bytes);
  auto layer_out = std::make_unique<std::byte[]>(max_layer_size_bytes);
  // NOTE there is an assumption here that output number is less or equal to input number for all layers

  memcpy(layer_in.get(), input_vec.data(), input_size_temp);

  int side_inputs_offset = 0;
  int lidx = 0;
  for (auto& layer_hash : hashes) {
    output_size = input_size_temp * layer_hash.output_size() / layer_hash.input_default_chunk_size();
    const int nof_hashes = input_size_temp / layer_hash.input_default_chunk_size();

    auto config = default_hash_config();
    config.batch = nof_hashes;
    layer_hash.hash(layer_in.get(), layer_hash.input_default_chunk_size(), config, layer_out.get());

    // copy output outputs to inputs before moving to the next layer
    memcpy(layer_in.get(), layer_out.get(), output_size);
    input_size_temp = output_size;

    // std::cout << "Layer" << lidx << " output:\t";
    // for (int i = 0; i < output_size; i++)
    // {
    //   std::cout << std::to_integer<int>(layer_out[i]) << "\t";
    // }
    // std::cout << '\n';
  }

  // Compare computed root with the tree's root
  auto [root, root_size] = tree.get_merkle_root();

  for (int i = 0; i < root_size; i++) {
    if (root[i] != layer_out[i]) { return false; }
  }
  return true;
}

/**
 * @brief Wrapper to the non-template version of is_valid_tree above, allowing to insert different types of arrays as
 * inputs. Builds tree in a straight-forward single-threaded manner and compares the result with Icicle's calculation.
 * @param tree - Merkle tree to test.
 * @param input_size - Size of input in bytes.
 * @param inputs - Input as a byte array.
 * @param hashes - Vector of hashes of each layer in the tree above.
 * @return True if the tree's calculations (icicle and test) match.
 */
template <typename T>
bool is_valid_tree(
  const MerkleTree& tree,
  int nof_inputs,
  const T* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config)
{
  return is_valid_tree(
    tree, nof_inputs * sizeof(T), sizeof(T), reinterpret_cast<const std::byte*>(inputs), hashes, config);
}

/**
 * @brief Function used by the HashApiTest to test the various Merkle trees defined in the tests below. Checks
 * validity of the tree construction, and correctness/incorrectness of valid/invalid proofs generated by the tree.
 * @param hashes - Vector of hashes of each layer in the tree above.
 * @param config - Merkle tree config (Mostly irrelevant for cpu tests).
 * @param output_store_min_layer - Store layer parameter for the Merkle tree builder.
 * @param nof_leaves - Size of the T leaves array.
 * @param leaves - Aforementioned leaves array.
 * @param explicit_leaf_size - Optional. Size of each leaf element in case that leaves is given as a byte array. NOTE 
 *                             test will fail if this value isn't default (1) and T != std::byte
 */
template <typename T>
void test_merkle_tree(
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config,
  const int output_store_min_layer,
  int nof_leaves,
  const T* leaves,
  unsigned explicit_leaf_size = 1)
{
  ASSERT_TRUE((explicit_leaf_size == 1 || std::is_same<T, std::byte>::value))
    << "Explicit leaf size should only be given when the given leaves array is a bytes array.";

  unsigned leaf_size = explicit_leaf_size > 1 ? explicit_leaf_size : sizeof(T);
  auto prover_tree = MerkleTree::create(
    hashes, leaf_size, output_store_min_layer); // TODO add config so padding can be checked at creation
  auto verifier_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);

  // assert that incorrect size fails
  if (config.padding_policy == PaddingPolicy::None) {
    ASSERT_NE(prover_tree.build(leaves, nof_leaves - 1, config), eIcicleError::SUCCESS);
    ASSERT_NE(prover_tree.build(leaves, nof_leaves + 1, config), eIcicleError::SUCCESS);
  }
  // build tree
  START_TIMER(MerkleTree_build)
  ICICLE_CHECK(prover_tree.build(leaves, nof_leaves * explicit_leaf_size, config));
  END_TIMER(MerkleTree_build, "Merkle Tree CPU", true)

  ASSERT_TRUE(is_valid_tree<T>(prover_tree, nof_leaves * explicit_leaf_size, leaves, hashes, config))
    << "Tree wasn't built correctly.";

  // Create wrong input leaves by taking the original input and swapping some leaves by random values
  auto wrong_leaves = std::make_unique<T[]>(nof_leaves * explicit_leaf_size);
  memcpy(wrong_leaves.get(), leaves, nof_leaves * explicit_leaf_size);
  const uint64_t nof_indices_modified = 5;
  unsigned int wrong_indices[nof_indices_modified];
  HashApiTest::randomize(wrong_indices, nof_indices_modified);
  for (int i = 0; i < nof_indices_modified; i++) {
    int wrong_index = wrong_indices[i] % (nof_leaves * explicit_leaf_size);
    int og_value = static_cast<int>(wrong_leaves[wrong_index]);
    wrong_leaves[wrong_index] = static_cast<T>(rand());
    int new_value = static_cast<int>(wrong_leaves[wrong_index]);
    ICICLE_LOG_DEBUG << "Wrong input is modified at index " << wrong_index << " from " << og_value << " to "
                     << new_value;
  }

  // Test the paths at the random indices (Both that the original input is valid and the modified input isn't)
  for (int i = 0; i < nof_indices_modified; i++) {
    int leaf_idx = (wrong_indices[i] % (nof_leaves * explicit_leaf_size)) / explicit_leaf_size;
    ICICLE_LOG_DEBUG << "Checking proof of index " << leaf_idx << " (Byte idx " << leaf_idx * explicit_leaf_size << ")";

    // get root and merkle-path for a leaf
    auto [root, root_size] = prover_tree.get_merkle_root();
    MerkleProof merkle_proof{};
    ICICLE_CHECK(
      prover_tree.get_merkle_proof(leaves, nof_leaves * explicit_leaf_size, leaf_idx, false, config, merkle_proof));

    // Test valid proof
    bool verification_valid = false;
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid) << "Proof of valid inputs at index " << leaf_idx
                                    << " is invalid (And should be valid).";

    // Test invalid proof (By modifying random data in the leaves)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(
      wrong_leaves.get(), nof_leaves * explicit_leaf_size, leaf_idx, false, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid) << "Proof of invalid inputs at index " << leaf_idx
                                     << " is valid (And should be invalid).";

    // Same for pruned proof
    verification_valid = false;
    ICICLE_CHECK(
      prover_tree.get_merkle_proof(leaves, nof_leaves * explicit_leaf_size, leaf_idx, true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid) << "Pruned proof of valid inputs at index " << leaf_idx
                                    << " is invalid (And should be valid).";

    // Test invalid proof (By modifying random data in the leaves)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(
      wrong_leaves.get(), nof_leaves * explicit_leaf_size, leaf_idx, true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid) << "Pruned proof of invalid inputs at index " << leaf_idx
                                     << " is valid (And should be invalid).";
  }
}

TEST_F(HashApiTest, MerkleTreeBasic)
{
  const int leaf_size = sizeof(uint32_t);
  const int nof_leaves = 50;
  uint32_t leaves[nof_leaves];
  randomize(leaves, nof_leaves);

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

  test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves, leaves);
}

TEST_F(HashApiTest, MerkleTreeZeroPadding)
{
  // const int leaf_size = sizeof(uint32_t); // 1
  // const int nof_leaves = 5;
  // uint32_t leaves[nof_leaves];
  // // randomize(leaves, nof_leaves);
  // for (int i = 0; i < nof_leaves; i++)
  // {
  //   leaves[i] = i;
  // }

  // auto layer0_hash = HashSumBackend::create(2 * leaf_size, leaf_size);  // in 2 leaves, out 1 leaves
  //                                                                       // 8B ->  4B -> 2B -> 1B

  // std::vector<Hash> hashes = {layer0_hash, layer0_hash, layer0_hash};
  const int leaf_size = sizeof(uint32_t);
  const int nof_leaves = 100;
  uint32_t leaves[nof_leaves];
  // randomize(leaves, nof_leaves);
  for (int i = 0; i < nof_leaves; i++) {
    leaves[i] = i + 1;
  }

  // define the merkle tree
  auto layer0_hash = HashSumBackend::create(5 * leaf_size, 2 * leaf_size); // in 5 leaves, out 2 leaves  200B ->  80B
  auto layer1_hash = HashSumBackend::create(4 * leaf_size, leaf_size);     // in 4 leaves, out 1 leaf    80B  ->  20B
  auto layer2_hash = HashSumBackend::create(leaf_size, leaf_size);         // in 1 leaf, out 1 leaf      20B  ->  20B
  auto layer3_hash = HashSumBackend::create(10 * leaf_size, leaf_size); // in 5 leaves, out 1 leaf    20B  ->  4B output

  int total_nof_input_hashes = nof_leaves * leaf_size / layer0_hash.input_default_chunk_size();
  std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
  int output_store_min_layer = 0;

  auto config = default_merkle_tree_config();
  // Test zero padding
  config.padding_policy = PaddingPolicy::ZeroPadding;

  // When tree is full (no padding required)
  // test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves, leaves);
  // When a whole number of hashes is missing - even number to ensure whole number of leaves
  // int nof_hashes = ((rand() % ((total_nof_input_hashes / 2) - 2)) + 1) * 2;
  // ICICLE_LOG_DEBUG << "Nof used hashes " << nof_hashes << " / " << total_nof_input_hashes;
  // test_merkle_tree(
  //   hashes, config, output_store_min_layer, nof_hashes * layer0_hash.input_default_chunk_size() / leaf_size, leaves);
  // // One leaf in tree
  // test_merkle_tree(hashes, config, output_store_min_layer, 1, leaves);

  // Lastly check case where last used hash isn't fully occupied
  auto byte_leaves = reinterpret_cast<const std::byte*>(leaves);
  int byte_size;
  do
  {
    byte_size = rand() % (nof_leaves * leaf_size);
  } while (byte_size % leaf_size == 0);
  byte_size = 327;
  ICICLE_LOG_DEBUG  << "Size of input in bytes: " << byte_size << "\t("
                    << float(byte_size) / leaf_size << " / " << nof_leaves << " leaves)";
  
  auto prover_tree = MerkleTree::create(
    hashes, leaf_size, output_store_min_layer); // TODO add config so padding can be checked at creation
  auto verifier_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);

  // build tree
  START_TIMER(MerkleTree_build)
  ICICLE_CHECK(prover_tree.build(byte_leaves, byte_size, config));
  END_TIMER(MerkleTree_build, "Merkle Tree CPU", true)

  ASSERT_TRUE(is_valid_tree(prover_tree, byte_size, byte_leaves, hashes, config))
    << "Tree wasn't built correctly.";

  auto wrong_bytes = std::make_unique<std::byte[]>(byte_size);
  memcpy(wrong_bytes.get(), byte_leaves, byte_size);
  // Modify the last byte as the only difference of this test from the previous is proof for the partial index
  wrong_bytes[byte_size - 1] = static_cast<std::byte>(rand());

  int leaf_idx = byte_size / leaf_size;
  ICICLE_LOG_DEBUG << "Checking proof of index " << leaf_idx << " (Byte idx " << leaf_idx * leaf_size << ")";

  // get root and merkle-path for a leaf
  auto [root, root_size] = prover_tree.get_merkle_root();
  MerkleProof merkle_proof{};
  ICICLE_CHECK(
    prover_tree.get_merkle_proof(byte_leaves, byte_size, leaf_idx, false, config, merkle_proof));

  // Test valid proof
  bool verification_valid = false;
  ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
  ASSERT_TRUE(verification_valid) << "Proof of valid inputs at index " << leaf_idx
                                  << " is invalid (And should be valid).";

  // Test invalid proof (By modifying random data in the leaves)
  verification_valid = true;
  ICICLE_CHECK(prover_tree.get_merkle_proof(wrong_bytes.get(), byte_size, leaf_idx, false, config, merkle_proof));
  ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
  ASSERT_FALSE(verification_valid) << "Proof of invalid inputs at index " << leaf_idx
                                    << " is valid (And should be invalid).";

  // Same for pruned proof
  verification_valid = false;
  ICICLE_CHECK(prover_tree.get_merkle_proof(byte_leaves, byte_size, leaf_idx, true, config, merkle_proof));
  ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
  ASSERT_TRUE(verification_valid) << "Pruned proof of valid inputs at index " << leaf_idx
                                  << " is invalid (And should be valid).";

  // Test invalid proof (By modifying random data in the leaves)
  verification_valid = true;
  ICICLE_CHECK(prover_tree.get_merkle_proof(wrong_bytes.get(), byte_size, leaf_idx, true, config, merkle_proof));
  ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
  ASSERT_FALSE(verification_valid) << "Pruned proof of invalid inputs at index " << leaf_idx
                                    << " is valid (And should be invalid).";
}

TEST_F(HashApiTest, MerkleTreeMixMediumSize)
{
  const uint32_t leaf_size = sizeof(uint32_t);
  const uint32_t total_input_size = (1 << 20);
  const uint32_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<uint32_t[]>(nof_leaves);
  randomize(leaves.get(), nof_leaves);

  // TODO repeat for all devices

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

  test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves, leaves.get());
}

TEST_F(HashApiTest, MerkleTreeLarge)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 28);
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), total_input_size);

  // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
  auto layer_hash = Keccak256::create(32 * 2);
  // Calculate the tree height (log2 of the number of leaves for a binary tree)
  const int tree_height = static_cast<int>(std::log2(nof_leaves));
  // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
  std::vector<Hash> hashes(tree_height, layer_hash);

  auto config = default_merkle_tree_config();

  test_merkle_tree(hashes, config, 0 /*output_store_min_layer*/, nof_leaves, leaves.get(), leaf_size);
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
