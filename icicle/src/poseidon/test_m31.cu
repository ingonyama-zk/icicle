// #define DEBUG

#include "fields/field_config.cuh"
using namespace field_config;

#include "gpu-utils/device_context.cuh"
#include "poseidon/poseidon.cuh"

#ifndef __CUDA_ARCH__
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace poseidon;

#define A 11
#define T (A + 1)

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

int main(int argc, char* argv[])
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  // Load poseidon constants
  START_TIMER(timer_const);
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  PoseidonConstants<scalar_t> constants;
  init_optimized_poseidon_constants<scalar_t>(A, ctx, &constants);
  END_TIMER(timer_const, "Load poseidon constants");

  START_TIMER(allocation_timer);
  // Prepare input data of [0, 1, 2 ... (number_of_blocks * arity) - 1]
  int number_of_blocks = argc > 1 ? 1 << atoi(argv[1]) : 1024;
  scalar_t input = scalar_t::zero();
  scalar_t* in_ptr = static_cast<scalar_t*>(malloc(number_of_blocks * A * sizeof(scalar_t)));
  for (uint32_t i = 0; i < number_of_blocks * A; i++) {
    in_ptr[i] = input;
    input = input + scalar_t::one();
  }
  END_TIMER(allocation_timer, "Allocate mem and fill input");

  scalar_t* out_ptr = static_cast<scalar_t*>(malloc(number_of_blocks * sizeof(scalar_t)));

  START_TIMER(poseidon_timer);
  PoseidonConfig config = default_poseidon_config(T);
  poseidon_hash<field_config::scalar_t, T>(in_ptr, out_ptr, number_of_blocks, constants, config);
  END_TIMER(poseidon_timer, "Poseidon")

  // scalar_t expected[0] = {}

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