#include "gpu-utils/device_context.cuh"
#include "extern.cu"

// #define DEBUG

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

int main(int argc, char* argv[])
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  START_TIMER(allocation_timer);
  // Prepare input data of [0, 1, 2 ... (number_of_blocks * input_block_size) - 1]
  int number_of_blocks = argc > 1 ? 1 << atoi(argv[1]) : 1024;
  int input_block_size = argc > 2 ? atoi(argv[2]) : 136;

  uint8_t* in_ptr = static_cast<uint8_t*>(malloc(number_of_blocks * input_block_size));
  for (uint64_t i = 0; i < number_of_blocks * input_block_size; i++) {
    in_ptr[i] = (uint8_t)i;
  }

  END_TIMER(allocation_timer, "Allocate mem and fill input");

  uint8_t* out_ptr = static_cast<uint8_t*>(malloc(number_of_blocks * (D / 8)));

  START_TIMER(keccak_timer);
  KeccakConfig config = default_keccak_config();
  keccak256_cuda(in_ptr, input_block_size, number_of_blocks, out_ptr, config);
  END_TIMER(keccak_timer, "Keccak")

  for (int i = 0; i < number_of_blocks; i++) {
#ifdef DEBUG
    uint8_to_hex_string(out_ptr + i * (D / 8), D / 8);
#endif
  }

  free(in_ptr);
  free(out_ptr);
}

#endif