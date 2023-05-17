#define DEBUG

#include "poseidon.cu"

#ifndef __CUDA_ARCH__
#include <iostream>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
  const int arity = 8;
  const int t = arity + 1;

  Poseidon poseidon(arity);

  int number_of_blocks = 1000000;

  scalar_t input = scalar_t::zero();
  scalar_t * in_ptr = static_cast< scalar_t * >(malloc(number_of_blocks * arity * sizeof(scalar_t)));
  for (uint32_t i = 0; i < number_of_blocks * arity; i++) {
    in_ptr[i] = input;
    input = input + scalar_t::one();
  }
  std::cout << std::endl;

  scalar_t * out_ptr = static_cast< scalar_t * >(malloc(number_of_blocks * sizeof(scalar_t)));

  auto start_time = std::chrono::high_resolution_clock::now();

  poseidon.hash_blocks(in_ptr, number_of_blocks, out_ptr, Poseidon::HashType::MerkleTree);

  #ifdef DEBUG
  for (int i = 0; i < number_of_blocks; i++) {
    //print_scalar_t(out_ptr[i]);
  }
  #endif

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

  free(in_ptr);
  free(out_ptr);
}

#endif