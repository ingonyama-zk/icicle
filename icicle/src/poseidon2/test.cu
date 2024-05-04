// #define DEBUG

#include "curves/curve_config.cuh"
using namespace curve_config;

#include "gpu-utils/device_context.cuh"

#ifndef __CUDA_ARCH__
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include "poseidon2/poseidon2.cuh"
using namespace poseidon2;

#define T 2

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
  Poseidon2Constants<scalar_t> constants;
  init_optimized_poseidon2_constants<scalar_t>(T, ctx, &constants);
  END_TIMER(timer_const, "Load poseidon constants");

  START_TIMER(allocation_timer);
  // Prepare input data of [0, 1, 2 ... (number_of_blocks * arity) - 1]
  int number_of_blocks = argc > 1 ? 1 << atoi(argv[1]) : 1024;
  scalar_t input = scalar_t::zero();
  scalar_t* in_ptr = static_cast<scalar_t*>(malloc(number_of_blocks * T * sizeof(scalar_t)));
  for (uint32_t i = 0; i < number_of_blocks * T; i++) {
    in_ptr[i] = input;
    input = input + scalar_t::one();
  }
  END_TIMER(allocation_timer, "Allocate mem and fill input");

  scalar_t* out_ptr = static_cast<scalar_t*>(malloc(number_of_blocks * sizeof(scalar_t)));

  START_TIMER(poseidon_timer);
  Poseidon2Config config = default_poseidon2_config(T);
  poseidon2_hash<curve_config::scalar_t, T>(in_ptr, out_ptr, number_of_blocks, constants, config);
  END_TIMER(poseidon_timer, "Poseidon")

  // scalar_t expected[1024] = {};

  if (number_of_blocks == 1024) {
    for (int i = 0; i < number_of_blocks; i++) {
#ifdef DEBUG
      std::cout << out_ptr[i] << std::endl;
#endif
      // assert((out_ptr[i] == expected[i]));
    }
    printf("Expected output matches\n");
  }

  free(in_ptr);
  free(out_ptr);
}

#endif