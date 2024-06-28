#ifndef __CUDA_ARCH__
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>

#include "merkle-tree/merkle.cuh"

#include "poseidon2/poseidon2.cuh"

#include "api/babybear.h"
using namespace babybear;

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

int main(int argc, char* argv[])
{
  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N elements
  uint32_t tree_arity = 2;
  uint32_t width = 16;
  uint32_t input_block_len = 8;
  uint32_t rate = 8;
  uint32_t digest_elements = 8;
  uint64_t tree_height = argc > 1 ? atoi(argv[1]) : 3;
  uint64_t number_of_leaves = pow(tree_arity, tree_height);
  uint64_t total_number_of_leaves = number_of_leaves * input_block_len;

  // Load poseidon constants
  START_TIMER(timer_const);
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  poseidon2::Poseidon2<scalar_t> poseidon(
    width, rate, poseidon2::MdsType::DEFAULT_MDS, poseidon2::DiffusionStrategy::DEFAULT_DIFFUSION, ctx);
  END_TIMER(timer_const, "Load poseidon constants");

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = argc > 2 ? atoi(argv[2]) : 3;
  size_t digests_len = merkle_tree::get_digests_len(keep_rows - 1, tree_arity, digest_elements);

  /// Fill leaves with scalars [0, 1, ... 2^tree_height - 1]
  START_TIMER(timer_allocation);
  scalar_t input = scalar_t::zero();

  unsigned int number_of_inputs = 1;
  Matrix<scalar_t>* leaves = static_cast<Matrix<scalar_t>*>(malloc(number_of_inputs * sizeof(Matrix<scalar_t>)));
  std::vector<scalar_t> matrix;
  matrix.reserve(total_number_of_leaves);

  for (uint64_t i = 0; i < total_number_of_leaves; i++) {
    matrix[i] = input;
    input = input + scalar_t::one();
  }

  leaves[0] = {
    matrix.data(),
    input_block_len,
    number_of_leaves,
  };

  END_TIMER(timer_allocation, "Allocated memory for leaves: ");

  /// Allocate memory for digests of {keep_rows} rows of a tree
  START_TIMER(timer_digests);
  size_t digests_mem = digests_len * sizeof(scalar_t);
  scalar_t* digests = static_cast<scalar_t*>(malloc(digests_mem));
  END_TIMER(timer_digests, "Allocated memory for digests");

  // std::cout << "Memory for leaves = " << total_number_of_leaves * sizeof(scalar_t) / 1024 / 1024 << " MB; " <<
  // leaves_mem / 1024 / 1024 / 1024 << " GB"
  //           << std::endl;
  std::cout << "Number of leaves = " << number_of_leaves << std::endl;
  std::cout << "Total Number of leaves = " << total_number_of_leaves << std::endl;
  std::cout << "Memory for digests = " << digests_mem / 1024 / 1024 << " MB; " << digests_mem / 1024 / 1024 / 1024
            << " GB" << std::endl;
  std::cout << "Number of digest elements = " << digests_len << std::endl;
  std::cout << std::endl;

  // std::cout << "Total RAM consumption = " << (digests_mem + leaves_mem) / 1024 / 1024 << " MB; "
  //           << (digests_mem + leaves_mem) / 1024 / 1024 / 1024 << " GB" << std::endl;

  merkle_tree::TreeBuilderConfig tree_config = merkle_tree::default_merkle_config();
  tree_config.arity = tree_arity;
  tree_config.keep_rows = keep_rows;
  tree_config.digest_elements = digest_elements;
  START_TIMER(timer_merkle);
  babybear_mmcs_commit_cuda(leaves, number_of_inputs, digests, &poseidon, &poseidon, tree_config);
  END_TIMER(timer_merkle, "Merkle tree built: ")

  for (int i = 0; i < digests_len; i++) {
    std::cout << digests[i] << std::endl;
  }

  // Use this to generate test vectors
  // for (int i = 0; i < digests_len; i++) {
  //   std::cout << "{";
  //   for (int j = 0; j < 8; j++) {
  //     std::cout << ((uint64_t*)&digests[i].limbs_storage)[j];
  //     if (j != 7) { std::cout << ", "; }
  //   }
  //   std::cout << "}," << std::endl;
  // }

  /// These scalars are digests of top-7 rows of a Merkle tree.
  /// Arity = 2, Tree height = 28, keep_rows = 7
  /// They are aligned in the following format:
  ///  L-7      L-6     L-5       L-4       L-3       L-2    L-1
  /// [0..63, 64..95, 96..111, 112..119, 120..123, 124..125, 126]
  scalar_t expected[0] = {};

  for (int i = 0; i < digests_len; i++) {
    scalar_t root = digests[i];
    assert(root == expected[i]);
  }
  free(digests);
  free(leaves);
}

#endif