#pragma once
#ifndef ERR_H
#define ERR_H

#include <iostream>

// TODO: ? do{..}while(0) as per https://hownot2code.wordpress.com/2016/12/05/do-while-0-in-macros/

#define CHECK_CUDA_ERROR(val)                                                                                          \
  do {                                                                                                                 \
    check((val), #val, __FILE__, __LINE__);                                                                            \
  } while (0)
template <typename T>
void inline check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
  }
}

#define CHECK_LAST_CUDA_ERROR()                                                                                        \
  / do                                                                                                                 \
  {                                                                                                                    \
    / checkLast(__FILE__, __LINE__);                                                                                   \
    /                                                                                                                  \
  }                                                                                                                    \
  while (0)
void inline checkLast(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

#define CHECK_SYNC_DEVICE_ERROR()                                                                                      \
  / do                                                                                                                 \
  {                                                                                                                    \
    / checkSyncDevice(__FILE__, __LINE__);                                                                             \
    /                                                                                                                  \
  }                                                                                                                    \
  while (0) // TODO: redundant ?
void inline checkSyncDevice(const char* const file, const int line)
{
  cudaError_t err{cudaDeviceSynchronize()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

#define CHECK_LAST_IS_STICKY_ERROR() checkLastStickyError(__FILE__, __LINE__)
cudaError_t inline checkLastStickyError(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    // check for sticky (unrecoverable) error where context is corrupted
    // and the only option is to restart process
    err = cudaDeviceSynchronize(); // TODO: only one call to cudaDeviceSynchronize seems sufficent according to
                                   // https://forums.developer.nvidia.com/t/cuda-errors-determine-sticky-ness/271625
    if (err != cudaSuccess) {      // we suspect sticky error, since it wasn't reset by cudaGetLastError
      // we are practically almost sure error is sticky
      std::cerr << "Please note the error is reported here and may be caused by prior calls" << std::endl;
      std::cerr << "!!!Unrecoverable CUDA Runtime Error detected at: " << file << ":" << line << std::endl;
      std::cerr << cudaGetErrorString(err) << std::endl;
      // TODO: common practice in C++ is to throw here instead
    }
  }

  return cudaSuccess; // TODO: or err? - returning cudaSuccess here shows it's non-sticky without using tuple
}

#endif
