#include "gpu-utils/device_context.cuh"
#include "merkle-tree/merkle.cuh"
#include "extern.cu"

#ifndef __CUDA_ARCH__
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace keccak;

#define D 256

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

void uint8_to_hex_string(const uint8_t* values, int size)
{
  std::stringstream ss;

  for (int i = 0; i < size; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)values[i];
  }

  std::string hexString = ss.str();
  std::cout << hexString << std::endl;
}

#define A 2

int main(int argc, char* argv[])
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N-1 elements
  uint32_t input_block_len = 136;
  uint32_t tree_height = argc > 1 ? atoi(argv[1]) : 10;
  uint32_t number_of_leaves = pow(A, tree_height);
  uint32_t total_number_of_leaves = number_of_leaves * input_block_len;

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = argc > 2 ? atoi(argv[2]) : 7;
  size_t digests_len = merkle_tree::get_digests_len(keep_rows - 1, A, 1);

  /// Fill leaves with scalars [0, 1, ... 2^tree_height - 1]
  START_TIMER(timer_allocation);
  uint8_t input = 0;
  uint8_t* leaves = static_cast<uint8_t*>(malloc(total_number_of_leaves));
  for (uint64_t i = 0; i < total_number_of_leaves; i++) {
    leaves[i] = (uint8_t)i;
  }
  END_TIMER(timer_allocation, "Allocated memory for leaves: ");

  /// Allocate memory for digests of {keep_rows} rows of a tree
  START_TIMER(timer_digests);
  size_t digests_mem = digests_len * sizeof(uint64_t);
  uint64_t* digests = static_cast<uint64_t*>(malloc(digests_mem));
  END_TIMER(timer_digests, "Allocated memory for digests");

  std::cout << "Memory for leaves = " << total_number_of_leaves / 1024 / 1024 << " MB; "
            << total_number_of_leaves / 1024 / 1024 / 1024 << " GB" << std::endl;
  std::cout << "Number of leaves = " << number_of_leaves << std::endl;
  std::cout << "Total Number of leaves = " << total_number_of_leaves << std::endl;
  std::cout << "Memory for digests = " << digests_mem / 1024 / 1024 << " MB; " << digests_mem / 1024 / 1024 / 1024
            << " GB" << std::endl;
  std::cout << "Number of digest elements = " << digests_len << std::endl;

  std::cout << "Total RAM consumption = " << (digests_mem + total_number_of_leaves) / 1024 / 1024 << " MB; "
            << (digests_mem + total_number_of_leaves) / 1024 / 1024 / 1024 << " GB" << std::endl;

  merkle_tree::TreeBuilderConfig config = merkle_tree::default_merkle_config();
  config.arity = A;
  config.keep_rows = keep_rows;
  START_TIMER(keccak_timer);
  build_keccak256_merkle_tree_cuda(leaves, digests, tree_height, input_block_len, config);
  END_TIMER(keccak_timer, "Keccak")

  for (int i = 0; i < digests_len; i++) {
    uint64_t root = digests[i];
    std::cout << root << std::endl;
    // assert(root == expected[i]);
  }
  free(digests);
  free(leaves);
}

#endif