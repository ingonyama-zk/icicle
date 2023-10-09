#pragma once
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

#define CHECK_SYNC_DEVICE_ERROR() syncDevice(__FILE__, __LINE__)
void syncDevice(const char* const file, const int line)
{
  cudaError_t err{cudaDeviceSynchronize()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

// Define ERROR_MODE
#define TEST 1
#define PRODUCTION 2

#define ERROR_MODE TEST // Set to either TEST or PRODUCTION

// Define CHECK_TEST and CHECK_PRODUCTION macros
#define CHECK_TEST CHECK_SYNC_DEVICE_ERROR
#define CHECK_PRODUCTION CHECK_LAST_CUDA_ERROR

// Define CHECK based on MODE
#if ERROR_MODE == TEST
    #define CHECK_CUDA_ERROR CHECK_TEST
#elif ERROR_MODE == PRODUCTION
    #define CHECK_CUDA_ERROR CHECK_PRODUCTION
#else
    #error "Invalid ERROR_MODE!"
#endif
