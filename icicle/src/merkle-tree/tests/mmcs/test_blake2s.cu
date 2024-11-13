#ifndef __CUDA_ARCH__
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>

#include "merkle-tree/merkle.cuh"
#include "hash/blake2s/blake2s.cuh"

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

using namespace blake2s;

int main(int argc, char* argv[])
{
  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N elements
  uint32_t tree_arity = 2;
  uint32_t input_block_len = 4;
  uint32_t rate = 16;
  uint32_t digest_elements = 32;
  uint32_t copied_matrices = 2;
  uint64_t tree_height = argc > 1 ? atoi(argv[1]) : 4;
  uint64_t number_of_leaves = pow(tree_arity, tree_height);
  uint64_t total_number_of_leaves = number_of_leaves * input_block_len;

  bool are_inputs_on_device = true;

  device_context::DeviceContext ctx = device_context::get_default_device_context();
  Blake2s hasher;

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = argc > 2 ? atoi(argv[2]) : tree_height + 1;
  size_t digests_len = merkle_tree::get_digests_len(keep_rows - 1, tree_arity, digest_elements);

  /// Fill leaves with scalars [0, 1, ... 2^tree_height - 1]
  START_TIMER(timer_allocation);
  // unsigned int number_of_inputs = tree_height * copied_matrices;
  unsigned int number_of_inputs = tree_height;
  Matrix<BYTE>* leaves = static_cast<Matrix<BYTE>*>(malloc(number_of_inputs * copied_matrices * sizeof(Matrix<BYTE>)));
  uint64_t current_matrix_rows = number_of_leaves;
  for (int i = 0; i < number_of_inputs; i++) {
    uint64_t current_matrix_size = current_matrix_rows * input_block_len;
    for (int j = 0; j < copied_matrices; j++) {
      BYTE* matrix = static_cast<BYTE*>(malloc(current_matrix_size));

      for (uint64_t k = 0; k < current_matrix_rows; k++) {
        ((uint32_t*)matrix)[k] = k;
      }

      BYTE* d_matrix;
      if (are_inputs_on_device) {
        cudaMalloc(&d_matrix, current_matrix_size);
        cudaMemcpy(d_matrix, matrix, current_matrix_size, cudaMemcpyHostToDevice);
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
  size_t digests_mem = digests_len;
  BYTE* digests = static_cast<BYTE*>(malloc(digests_mem));

  std::cout << "Number of leaves = " << number_of_leaves << std::endl;
  std::cout << "Total Number of leaves = " << total_number_of_leaves << std::endl;
  std::cout << "Memory for digests = " << digests_mem / 1024 / 1024 << " MB; " << digests_mem / 1024 / 1024 / 1024
            << " GB" << std::endl;
  std::cout << "Number of digest elements = " << digests_len << std::endl;
  std::cout << std::endl;

  merkle_tree::TreeBuilderConfig tree_config = merkle_tree::default_merkle_config();
  tree_config.are_inputs_on_device = are_inputs_on_device;
  tree_config.arity = tree_arity;
  tree_config.keep_rows = keep_rows;
  tree_config.digest_elements = digest_elements;
  START_TIMER(timer_merkle);
  blake2s_mmcs_commit_cuda(leaves, number_of_inputs * copied_matrices, digests, &hasher, tree_config);
  END_TIMER(timer_merkle, "Merkle tree built: ")

  for (int i = 0; i < digests_len; i++) {
    if (i % 32 == 0) { std::cout << std::endl; }
    printf("%.2X", digests[i]);
  }
  free(digests);
  free(leaves);
}

#endif