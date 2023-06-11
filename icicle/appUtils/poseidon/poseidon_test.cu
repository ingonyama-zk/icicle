#define DEBUG

#include "../../curves/bls12_381/curve_config.cuh"
#include "../../curves/bls12_381/poseidon.cu"

#ifndef __CUDA_ARCH__
#include <iostream>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
  const int arity = 2;
  const int t = arity + 1;

  Poseidon<BLS12_381::scalar_t> poseidon(arity);

  int number_of_blocks = 1024;

  BLS12_381::scalar_t input = BLS12_381::scalar_t::zero();
  BLS12_381::scalar_t * in_ptr = static_cast< BLS12_381::scalar_t * >(malloc(number_of_blocks * arity * sizeof(BLS12_381::scalar_t)));
  for (uint32_t i = 0; i < number_of_blocks * arity; i++) {
    in_ptr[i] = input;
    input = input + BLS12_381::scalar_t::one();
  }
  std::cout << std::endl;

  BLS12_381::scalar_t * out_ptr = static_cast< BLS12_381::scalar_t * >(malloc(number_of_blocks * sizeof(BLS12_381::scalar_t)));

  auto start_time = std::chrono::high_resolution_clock::now();

  poseidon.hash_blocks(in_ptr, number_of_blocks, out_ptr, Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree);

  #ifdef DEBUG
  for (int i = 0; i < number_of_blocks; i++) {
    std::cout << out_ptr[i] << std::endl;
  }
  #endif

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

  free(in_ptr);
  free(out_ptr);
}

#endif