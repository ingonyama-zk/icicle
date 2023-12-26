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

#define CHK_OK(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
cudaError_t inline check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
  }

  return err;
}

#define CHK_LAST() checkLast(__FILE__, __LINE__)
cudaError_t inline checkLast(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  return err;
}

// TODO: one macro that optionally (by compile-time switch) doesn't throw 
#define CHK_CUDA_NO_THROW() checkLastStickyError(__FILE__, __LINE__, false) 

#define CHK_CUDA() checkLastStickyError(__FILE__, __LINE__)

cudaError_t inline checkLastStickyError(const char* const file, const int line, bool isThrowing = true)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    // check for sticky (unrecoverable) error when the only option is to restart process
    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) { // we suspect sticky error
      // we are practically almost sure error is sticky
      if (isThrowing) {
        // TODO: fmt::format introduced only in C++20
        std::string err2_msg = "!!!Unrecoverable!!! : " + std::string{cudaGetErrorString(err2)} +
                               " : detected at : " + std::string(file) + ":" + std::to_string(line) +
                               "\nThe error is reported there and may be caused by prior calls.";
        std::cerr << err2_msg << std::endl; // TODO: Logging
        throw IcicleError{err2, err2_msg};
      } else {
        err = err2;
      }
    }
  }

  return err;
}

#endif
