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
    for (int i = 0; i < (size * sizeof(T) / sizeof(uint32_t)); ++i) {
      u32_arr[i] = dist(gen);
    }
  }
};

TEST_F(HashApiTest, Keccak256)
{
  const uint64_t input_size = 64; // Number of input elements
  const uint64_t output_size = 256;

  // Create unique pointers for input and output arrays
  auto input = std::make_unique<uint32_t[]>(input_size);
  auto output = std::make_unique<std::byte[]>(output_size);

  // Randomize the input array
  randomize(input.get(), input_size);

  auto config = default_hash_config();
  // Create Keccak-256 hash object
  auto keccak256 = Keccak256::create();
  // Run single hash operation
  ICICLE_CHECK(keccak256.hash(input.get(), input_size, config, output.get()));

  // Batch hash (hashing each half separately)
  auto output_batch = std::make_unique<std::byte[]>(output_size * 2);
  config.batch = 2;
  ICICLE_CHECK(keccak256.hash(input.get(), input_size / 2, config, output_batch.get()));
  // TODO: Verify output (e.g., check CPU against CUDA)
}

// TODO: add tests for all hashes

/****************************** Merkle **********************************/
typedef uint32_t merkel_hash_t;

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
    const merkel_hash_t* input_u32 = (const merkel_hash_t*)input;
    merkel_hash_t* output_u32 = (merkel_hash_t*)output;

    output_u32[0] = 0;
    int t_size = sizeof(merkel_hash_t);
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
  int nof_input_byes,
  const std::byte* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config
)
{ 
  int nof_outputs = nof_input_byes * hashes[0].output_size() / hashes[0].input_default_chunk_size();
  std::byte* layer_in =  new std::byte[nof_input_byes]; // Going layer by layer - having the input layer as the largest 
  std::byte* layer_out = new std::byte[nof_outputs];    // ensures these are the maximum sizes required for the arrays
  // NOTE there is an assumption here that output number is less or equal to input number for all layers

  for (int i = 0; i < nof_input_byes; i++) { layer_in[i] = inputs[i]; }
  
  // int nof_hashes = nof_inputs / hashes[0]->m_total_input_limbs;
  int side_inputs_offset = 0;
  for (int i = 0; i < hashes.size(); i++)
  {
    nof_outputs = nof_input_byes * hashes[i].output_size() / hashes[i].input_default_chunk_size();
    const int nof_hashes = nof_input_byes / hashes[i].input_default_chunk_size();

    // TODO actual config to allow gpu test as well?
    auto config = default_hash_config();
    config.batch = nof_hashes;
    hashes[i].hash(layer_in, hashes[i].input_default_chunk_size(), config, layer_out);

    // Transfer outputs to inputs before moving to the next layer
    for (int i = 0; i < nof_outputs; i++) { layer_in[i] = layer_out[i]; }
    nof_input_byes = nof_outputs;
  }

  // Compare computed root with the tree's root
  auto [root, root_size] = tree.get_merkle_root();

  for (int i = 0; i < root_size; i++) { ICICLE_ASSERT(root[i] == layer_out[i]) << "Mismatch in root[" << i << "]"; }

  delete[] layer_in;
  delete[] layer_out;
}

template<typename in_limb>
void assert_valid_tree(
  const MerkleTree& tree,
  int nof_inputs,
  const in_limb* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config
)
{
  return 
    assert_valid_tree(tree, nof_inputs * sizeof(in_limb), reinterpret_cast<const std::byte*>(inputs), hashes, config);
}

TEST_F(HashApiTest, MerkleTreeBasic)
{
  // define input
  int leaf_size = sizeof(merkel_hash_t);
  constexpr int nof_leaves = 50;
  merkel_hash_t leaves[nof_leaves];
  randomize(leaves, nof_leaves);

  merkel_hash_t other_leaves[nof_leaves];
  randomize(other_leaves, nof_leaves);
  
  // define the merkle tree
  auto config = default_merkle_tree_config();
  auto layer0_hash = HashSumBackend::create(5*leaf_size, 2*leaf_size);  // in 5 leaves, out 2 leaves  200B ->  80B
  auto layer1_hash = HashSumBackend::create(4*leaf_size, leaf_size);    // in 4 leaves, out 1 leaf    80B  ->  20B
  auto layer2_hash = HashSumBackend::create(leaf_size, leaf_size);      // in 1 leaf, out 1 leaf      20B  ->  20B
  auto layer3_hash = HashSumBackend::create(5*leaf_size, leaf_size);    // in 5 leaves, out 1 leaf    20B  ->  4B output
  
  std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
  
  int output_store_min_layer;
  randomize(&output_store_min_layer, 1);
  output_store_min_layer = output_store_min_layer & 3; // Ensure index is in a valid 0-3 range
  ICICLE_LOG_DEBUG << "Min store layer:\t" << output_store_min_layer;

  auto prover_tree =
    MerkleTree::create(hashes, sizeof(merkel_hash_t), output_store_min_layer);

  auto verifier_tree =
    MerkleTree::create(hashes, sizeof(merkel_hash_t), output_store_min_layer);

  // build tree

  //START_TIMER(MerkleTree_build)
  ICICLE_CHECK(prover_tree.build(leaves, nof_leaves, config));
  assert_valid_tree<uint32_t>(prover_tree, nof_leaves, leaves, hashes, config);
  //std::cout << "MerkleTree build time: ";
  //END_TIMER(MerkleTree_build, std::cout, true)

  // get root and merkle-path to an element
  int nof_idx_to_check = 5;
  uint64_t leaf_indices[nof_idx_to_check];
  randomize(leaf_indices, nof_idx_to_check);

  for (int i = 0; i < nof_idx_to_check; i++)
  {
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
    ICICLE_CHECK(prover_tree.get_merkle_proof(other_leaves, nof_leaves, leaf_idx, false, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid);

    // Same for pruned proof
    verification_valid = false;
    ICICLE_CHECK(prover_tree.get_merkle_proof(leaves, nof_leaves, leaf_idx, true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid);
    
    // Test invalid proof (By adding random data to the proof)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(other_leaves, nof_leaves, leaf_idx, true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid);
  }
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