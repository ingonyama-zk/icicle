#define DEBUG
#define MERKLE_DEBUG
#include "../../curves/bls12_381/curve_config.cuh"
#include "../../curves/bls12_381/merkle.cu"

#ifndef __CUDA_ARCH__
#include <iostream>
#include <chrono>
#include <fstream>
#include <math.h>

void dump(BLS12_381::scalar_t * buf, size_t size, std::string filepath) {
  std::ofstream outfile(filepath, std::ios::binary);
  for (int i = 0; i < size; i++) {
    BLS12_381::scalar_t el = buf[i];
    outfile.write((char *)&el, sizeof(BLS12_381::scalar_t));
  }
  outfile.close();
}

int main(int argc, char* argv[]) {
  using FpMilliseconds = 
    std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = 
    std::chrono::duration<float, std::chrono::microseconds::period>;
  
  const int arity = 2;
  const int t = arity + 1;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start_event, end_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);
  cudaEventRecord(start_event, stream);
  auto start_time1 = std::chrono::high_resolution_clock::now();
  Poseidon<BLS12_381::scalar_t> poseidon(arity, stream);

  auto end_time1 = std::chrono::high_resolution_clock::now();
  auto elapsed_time1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time1 - start_time1);
  printf("Elapsed time poseidon: %.0f us\n", FpMicroseconds(elapsed_time1).count());

  uint32_t tree_height = 30;
  uint32_t number_of_leaves = pow(arity, (tree_height - 1));
  uint32_t cutoff_height = 7;
  size_t cutoff_leaves_len = pow(arity, (cutoff_height - 1));
  std::cout << "Cutoff leaves = " << cutoff_leaves_len << std::endl;
  size_t cutoff_digests_len = get_digests_len(cutoff_height, arity);
  std::cout << "Cutoff digests = " << cutoff_digests_len << std::endl;

  auto start_time2 = std::chrono::high_resolution_clock::now();
  BLS12_381::scalar_t input = BLS12_381::scalar_t::zero();
  BLS12_381::scalar_t * leaves = static_cast< BLS12_381::scalar_t * >(malloc(number_of_leaves * sizeof(BLS12_381::scalar_t)));
  // memset(leaves, 1, number_of_leaves * sizeof(BLS12_381::scalar_t));
  for (uint32_t i = 0; i < number_of_leaves; i++) {
    leaves[i] = input;
    input = input + BLS12_381::scalar_t::one();
  }

  std::cout << "INPUT:" << std::endl;
  for (uint32_t i = 0; i < 10; i++) {
    std::cout << leaves[number_of_leaves - 1 - i] << std::endl;
  }

  auto end_time2 = std::chrono::high_resolution_clock::now();
  auto elapsed_time2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);
  printf("Leaves allocation: %.0f us\n", FpMicroseconds(elapsed_time2).count());

  size_t leaves_mem = number_of_leaves * sizeof(BLS12_381::scalar_t);
  std::cout << "Memory for leaves = " << leaves_mem / 1024 / 1024 << " MB; " << leaves_mem / 1024 / 1024 / 1024 << " GB" << std::endl;
  std::cout << "Number of leaves = " << number_of_leaves << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  auto digests_len = get_digests_len(tree_height, arity);
  BLS12_381::scalar_t * digests = static_cast< BLS12_381::scalar_t * >(malloc(digests_len * sizeof(BLS12_381::scalar_t)));
  // memset(digests, 0, digests_len * sizeof(BLS12_381::scalar_t));
  size_t digests_mem = digests_len * sizeof(BLS12_381::scalar_t);
  std::cout << "Memory for digests = " << digests_mem / 1024 / 1024 << " MB; " << digests_mem / 1024 / 1024 / 1024 << " GB" << std::endl;
  std::cout << "Number of digest elements = " << digests_len << std::endl;

  std::cout << "Total RAM consumption = " << (digests_mem + leaves_mem) / 1024 / 1024 << " MB; " << (digests_mem + leaves_mem) / 1024 / 1024 / 1024 << " GB" << std::endl;
  build_merkle_tree<BLS12_381::scalar_t>(leaves, digests, tree_height, poseidon, stream);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  auto tree_time = FpMicroseconds(elapsed_time).count();
  printf("Elapsed time in merkle tree building: %.0f us, %.0f ms, %.2f s\n", tree_time, tree_time / 1000, tree_time / 1000 / 1000);
  cudaEventRecord(end_event, stream);
  cudaEventSynchronize(end_event);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start_event, end_event);
  printf("Elapsed time: %8.3f ms\n", elapsedTime);

  // for (int i = 0; i < digests_len; i++) {
  //   std::cout << i << ": " << digests[i] << std::endl;
  // }
  // std::cout << std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << digests[digests_len - cutoff_digests_len - 1 - i] << std::endl;
  }

  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);
  free(leaves);
}

#endif