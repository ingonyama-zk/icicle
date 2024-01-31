#include <chrono>
#include <fstream>
#include <iostream>

// select the curve (only 2 available so far)
#define CURVE_ID 2
#include "curves/curve_config.cuh"
#include "utils/device_context.cuh"
// include Poseidon template
#include "appUtils/poseidon/poseidon.cu"
using namespace poseidon;
using namespace curve_config;

#define A 2
#define T A + 1

device_context::DeviceContext ctx= device_context::get_default_device_context();

// location of a tree node in the array for a given level and offset
inline uint32_t tree_index(uint32_t level, uint32_t offset) { return (1 << level) - 1 + offset; }

// We assume the tree has leaves already set, compute all other levels
// void build_tree(
//   const uint32_t tree_height, scalar_t* tree, Poseidon<BLS12_381::scalar_t>& poseidon, cudaStream_t stream)
// {
//   for (uint32_t level = tree_height - 1; level > 0; level--) {
//     const uint32_t next_level = level - 1;
//     const uint32_t next_level_width = 1 << next_level;
//     poseidon.hash_blocks(
//       &tree[tree_index(level, 0)], next_level_width, &tree[tree_index(next_level, 0)],
//       Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree, stream);
//   }
// }

// search leaves for a given hash, return offset
// uint32_t query_membership(BLS12_381::scalar_t query, BLS12_381::scalar_t* tree, const uint32_t tree_height)
// {
//   const uint32_t tree_width = (1 << (tree_height - 1));
//   for (uint32_t i = 0; i < tree_width; i++) {
//     const BLS12_381::scalar_t leaf = tree[tree_index(tree_height - 1, i)];
//     if (leaf == query) {
//       return i; // found the hash
//     }
//   }
//   return tree_height; // hash not found
// }

// void generate_proof(
//   uint32_t position,
//   BLS12_381::scalar_t* tree,
//   const uint32_t tree_height,
//   uint32_t* proof_lr,
//   BLS12_381::scalar_t* proof_hash)
// {
//   uint32_t level_index = position;
//   for (uint32_t level = tree_height - 1; level > 0; level--) {
//     uint32_t lr;
//     uint32_t neighbour_index;
//     lr = level_index % 2;
//     if (lr == 0) {
//       // left
//       neighbour_index = level_index + 1;
//     } else {
//       // right
//       neighbour_index = level_index - 1;
//     }
//     proof_lr[level] = lr;
//     proof_hash[level] = tree[tree_index(level, neighbour_index)];
//     level_index /= 2;
//   }
//   // the proof must match this:
//   proof_hash[0] = tree[tree_index(0, 0)];
// }

// uint32_t validate_proof(
//   const BLS12_381::scalar_t hash,
//   const uint32_t tree_height,
//   const uint32_t* proof_lr,
//   const BLS12_381::scalar_t* proof_hash,
//   Poseidon<BLS12_381::scalar_t>& poseidon,
//   cudaStream_t stream)
// {
//   BLS12_381::scalar_t hashes_in[2], hash_out[1], level_hash;
//   level_hash = hash;
//   for (uint32_t level = tree_height - 1; level > 0; level--) {
//     if (proof_lr[level] == 0) {
//       hashes_in[0] = level_hash;
//       hashes_in[1] = proof_hash[level];
//     } else {
//       hashes_in[0] = proof_hash[level];
//       hashes_in[1] = level_hash;
//     }
//     // next level hash
//     poseidon.hash_blocks(hashes_in, 1, hash_out, Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree, stream);
//     level_hash = hash_out[0];
//   }
//   return proof_hash[0] == level_hash;
// }

int main(int argc, char* argv[])
{
  std::cout << "1. Defining the size of the example: height of the full binary Merkle tree" << std::endl;
  const uint32_t tree_height = 21;
  std::cout << "Tree height: " << tree_height << std::endl;
  const uint32_t tree_arity = 2;
  const uint32_t leaf_level = tree_height - 1;
  const uint32_t tree_width = 1 << leaf_level;
  std::cout << "Tree width: " << tree_width << std::endl;
  const uint32_t tree_size = (1 << tree_height) - 1;
  std::cout << "Tree size: " << tree_size << std::endl;
  scalar_t* tree = static_cast<scalar_t*>(malloc(tree_size * sizeof(scalar_t)));

  std::cout << "2. Hashing blocks in parallel" << std::endl;
  
  // ctx = device_context::get_default_device_context();
  init_optimized_poseidon_constants<scalar_t, T>(ctx);

  const uint32_t data_arity = 2;
  std::cout << "Block size (arity): " << data_arity << std::endl;
  std::cout << "Initializing blocks..." << std::endl;
  scalar_t d = scalar_t::zero();
  scalar_t* data = static_cast<scalar_t*>(malloc(tree_width * data_arity * sizeof(scalar_t)));

  for (uint32_t i = 0; i < tree_width * data_arity; i++) {
    data[i] = d;
    d = d + scalar_t::one();
  }

  std::cout << "Hashing blocks into tree leaves..." << std::endl;
  PoseidonConfig config = default_poseidon_config(T);
  poseidon_hash<curve_config::scalar_t, T>(data, &tree[tree_index(leaf_level, 0)], tree_width, preloaded_constants<curve_config::scalar_t, T>, config);
  // data_poseidon.hash_blocks(data, tree_width, &tree[tree_index(leaf_level, 0)], Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree, stream);

  // std::cout << "3. Building Merkle tree" << std::endl;
  // Poseidon<BLS12_381::scalar_t> tree_poseidon(tree_arity, stream);
  // build_tree(tree_height, tree, tree_poseidon, stream);

  // std::cout << "4. Generate membership proof" << std::endl;
  // uint32_t position = tree_width - 1;
  // std::cout << "Using the hash for block: " << position << std::endl;
  // BLS12_381::scalar_t query = tree[tree_index(leaf_level, position)];
  // uint32_t query_position = query_membership(query, tree, tree_height);
  // // allocate arrays for the proof
  // uint32_t* proof_lr = static_cast<uint32_t*>(malloc(tree_height * sizeof(uint32_t)));
  // BLS12_381::scalar_t* proof_hash =
  //   static_cast<BLS12_381::scalar_t*>(malloc(tree_height * sizeof(BLS12_381::scalar_t)));
  // generate_proof(query_position, tree, tree_height, proof_lr, proof_hash);

  // std::cout << "5. Validate the hash membership" << std::endl;
  // uint32_t validated;
  // const BLS12_381::scalar_t hash = tree[tree_index(leaf_level, query_position)];
  // validated = validate_proof(hash, tree_height, proof_lr, proof_hash, tree_poseidon, stream);
  // std::cout << "Validated: " << validated << std::endl;

  // std::cout << "6. Tamper the hash" << std::endl;
  // const BLS12_381::scalar_t tampered_hash = hash + BLS12_381::scalar_t::one();
  // validated = validate_proof(tampered_hash, tree_height, proof_lr, proof_hash, tree_poseidon, stream);
  // std::cout << "7. Invalidate tamper hash membership" << std::endl;
  // std::cout << "Validated: " << validated << std::endl;
  return 0;
}
