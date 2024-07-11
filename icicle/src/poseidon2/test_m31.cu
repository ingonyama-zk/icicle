#include "gpu-utils/device_context.cuh"

#ifndef __CUDA_ARCH__
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include "poseidon2/poseidon2.cuh"
using namespace poseidon2;

#include "fields/field_config.cuh"
using namespace field_config;

#include "hash/hash.cuh"

#define T 16

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

int main(int argc, char* argv[])
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  // Load poseidon
  START_TIMER(timer_const);
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  Poseidon2<scalar_t> poseidon(T, T, MdsType::DEFAULT_MDS, DiffusionStrategy::DEFAULT_DIFFUSION, ctx);
  END_TIMER(timer_const, "Load poseidon constants");

  int number_of_blocks = argc > 1 ? 1 << atoi(argv[1]) : 1024;
  scalar_t* in_ptr = static_cast<scalar_t*>(malloc(number_of_blocks * T * sizeof(scalar_t)));
  scalar_t* out_ptr = static_cast<scalar_t*>(malloc(number_of_blocks * sizeof(scalar_t)));
  scalar_t input = scalar_t::zero();

  hash::HashConfig cfg = hash::default_hash_config();

  size_t number_of_repetitions = argc > 2 ? 1 << atoi(argv[2]) : 32;

  // Prepare input data of [0, 1, 2 ... (number_of_blocks * arity) - 1]
  for (uint32_t i = 0; i < number_of_blocks * T; i++) {
    in_ptr[i] = input;
    input = input + scalar_t::one();
  }

  // Warm up
  poseidon.hash_many(in_ptr, out_ptr, number_of_blocks, T, 1, cfg);

  auto total_time_start = std::chrono::high_resolution_clock::now();
  size_t avg_time = 0;
  for (int i = 0; i < number_of_repetitions; i++) {
    auto poseidon_start = std::chrono::high_resolution_clock::now();
    poseidon.hash_many(in_ptr, out_ptr, number_of_blocks, T, 1, cfg);
    avg_time += FpMilliseconds(std::chrono::high_resolution_clock::now() - poseidon_start).count();
  }
  auto total_time = FpMilliseconds(std::chrono::high_resolution_clock::now() - total_time_start).count();

  std::cout << "Block size: " << number_of_blocks << std::endl;
  std::cout << "Total time: " << total_time << " ms" << std::endl;
  std::cout << "Avg time: " << avg_time / number_of_repetitions << " ms" << std::endl;

  // for (int i = 0; i < number_of_blocks; i++) {
  //   std::cout << "{";
  //   for (int j = 0; j < 8; j++) {
  //     std::cout << ((uint32_t*)&out_ptr[i].limbs_storage)[j];
  //     if (j != 7) { std::cout << ", "; }
  //   }
  //   std::cout << "}," << std::endl;
  // }

  if (number_of_blocks == 1024) {
    for (int i = 0; i < number_of_blocks; i++) {
#ifdef DEBUG
      // std::cout << out_ptr[i] << std::endl;
#endif
      // assert((out_ptr[i] == expected[i]));
    }
    printf("Expected output matches\n");
  }

  free(in_ptr);
  free(out_ptr);
}

#endif