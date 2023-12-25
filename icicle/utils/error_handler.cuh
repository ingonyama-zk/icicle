#pragma once
#ifndef ERR_H
#define ERR_H

#include <iostream>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

enum class IcicleError_t {
  UndefinedError = 0 // Assigning 0 as the value for UndefinedError
};

std::string inline IcicleGetErrorString(IcicleError_t error)
{
  switch (error) {
  case IcicleError_t::UndefinedError:
    return "Undefined error occurred.";
  default:
    return "Unknown error code.";
  }
}

class IcicleError : public std::runtime_error
{
private:
  int errCode; // Field to store the error code

public:
  // Constructor for cudaError_t with optional message
  IcicleError(cudaError_t cudaError, const std::string& msg = "")
      : std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(cudaError)) + " " + msg),
        errCode(static_cast<int>(cudaError))
  {
  }

  // Constructor for cudaError_t with const char* message
  IcicleError(cudaError_t cudaError, const char* msg) : IcicleError(cudaError, std::string(msg)) {}

  // Constructor for IcicleError_t with optional message
  IcicleError(IcicleError_t icicleError, const std::string& msg = "")
      : std::runtime_error("Icicle Error: " + IcicleGetErrorString(icicleError) + " " + msg),
        errCode(static_cast<int>(icicleError))
  {
  }

  // Constructor for IcicleError_t with const char* message
  IcicleError(IcicleError_t icicleError, const char* msg) : IcicleError(icicleError, std::string(msg)) {}

  // Getter for errCode
  int getErrorCode() const { return errCode; }
};

// TODO: ? do{..}while(0) as per https://hownot2code.wordpress.com/2016/12/05/do-while-0-in-macros/

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void inline check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void inline checkLast(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
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
    // check for sticky (unrecoverable) error when the only option is to restart process
    cudaError_t err2 =
      cudaDeviceSynchronize(); // TODO: only one call to cudaDeviceSynchronize seems sufficent according to
                               // https://forums.developer.nvidia.com/t/cuda-errors-determine-sticky-ness/271625
    if (err2 != cudaSuccess) { // we suspect sticky error
      // we are practically almost sure error is sticky
      // TODO: fmt::format introduced only in C++20
      std::string err2_msg = std::string{"Please note the error is reported here and may be caused by prior calls. "} +
                             std::string{cudaGetErrorString(err2)} +
                             std::string{"!!!Unrecoverable!!! CUDA Runtime Error detected at: "} + std::string(file) +
                             std::string(":") + std::to_string(line);
      std::cerr << err2_msg << std::endl; // TODO: Logging
      throw IcicleError{err2, err2_msg};
    }
  }

  return err;
}

#endif
