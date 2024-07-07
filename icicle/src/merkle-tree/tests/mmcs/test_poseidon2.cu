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
  uint32_t input_block_len = 600;
  uint32_t rate = 8;
  uint32_t digest_elements = 8;
  uint32_t copied_matrices = 1;
  uint64_t tree_height = argc > 1 ? atoi(argv[1]) : 3;
  uint64_t number_of_leaves = pow(tree_arity, tree_height);
  uint64_t total_number_of_leaves = number_of_leaves * input_block_len;

  bool are_inputs_on_device = true;

  // Load poseidon constants
  START_TIMER(timer_const);
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  poseidon2::Poseidon2<scalar_t> poseidon(
    width, rate, poseidon2::MdsType::PLONKY, poseidon2::DiffusionStrategy::MONTGOMERY, ctx);
  END_TIMER(timer_const, "Load poseidon constants");

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = argc > 2 ? atoi(argv[2]) : 3;
  size_t digests_len = merkle_tree::get_digests_len(keep_rows - 1, tree_arity, digest_elements);

  /// Fill leaves with scalars [0, 1, ... 2^tree_height - 1]
  START_TIMER(timer_allocation);
  scalar_t input = scalar_t::zero();

  // unsigned int number_of_inputs = tree_height * copied_matrices;
  unsigned int number_of_inputs = 1;
  Matrix<scalar_t>* leaves = static_cast<Matrix<scalar_t>*>(malloc(number_of_inputs * sizeof(Matrix<scalar_t>)));
  uint64_t current_matrix_rows = number_of_leaves;
  for (int i = 0; i < number_of_inputs; i++) {
    uint64_t current_matrix_size = current_matrix_rows * input_block_len;
    for (int j = 0; j < copied_matrices; j++) {
      scalar_t* matrix = static_cast<scalar_t*>(malloc(current_matrix_size * sizeof(scalar_t)));

      for (uint64_t k = 0; k < current_matrix_size; k++) {
        matrix[k] = input;
        input = input + scalar_t::one();
      }

      scalar_t* d_matrix;
      if (are_inputs_on_device) {
        cudaMalloc(&d_matrix, current_matrix_size * sizeof(scalar_t));
        cudaMemcpy(d_matrix, matrix, current_matrix_size * sizeof(scalar_t), cudaMemcpyHostToDevice);
      }

      leaves[i * copied_matrices + j] = {
        are_inputs_on_device ? d_matrix : matrix,
        input_block_len,
        current_matrix_rows,
      };
    }

    current_matrix_rows /= tree_arity;
  }

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
  tree_config.are_inputs_on_device = are_inputs_on_device;
  tree_config.arity = tree_arity;
  tree_config.keep_rows = keep_rows;
  tree_config.digest_elements = digest_elements;
  START_TIMER(timer_merkle);
  babybear_mmcs_commit_cuda(leaves, number_of_inputs, digests, &poseidon, &poseidon, tree_config);
  END_TIMER(timer_merkle, "Merkle tree built: ")

  for (int i = 0; i < 10; i++) {
    std::cout << digests[digests_len - i - 1] << std::endl;
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
    // assert(root == expected[i]);
  }
  free(digests);
  free(leaves);
}

#endif