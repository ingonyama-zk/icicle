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

#define CHK_ERR(err, func, file, line) check(err, func, file, line)
#define CHK_VAL(val, file, line)       check((val), #val, file, line)

cudaError_t inline check(cudaError_t err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error by: " << func << " at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl << std::endl;
  }

  return err;
}

// TODO: one macro that optionally (by compile-time switch) doesn't throw
#define CHK_STICKY_NO_THROW(val) checkCudaErrorIsSticky((val), #val, __FILE__, __LINE__, false)

#define CHK_LAST_STICKY_NO_THROW()                                                                                     \
  checkCudaErrorIsSticky(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__, false)

#define CHK_LAST() checkCudaErrorIsSticky(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__)

#define CHK_STICKY(val) checkCudaErrorIsSticky((val), #val, __FILE__, __LINE__)

cudaError_t inline checkCudaErrorIsSticky(
  cudaError_t err, const char* const func, const char* const file, const int line, bool isThrowing = true)
{
  if (err != cudaSuccess) {
    // check for sticky (unrecoverable) error when the only option is to restart process
    cudaError_t err2 = cudaDeviceSynchronize();
    if (err != err2) CHK_ERR(err, func, file, line);
    if (err2 != cudaSuccess) { // we suspect sticky error
      // we are practically almost sure error is sticky
      if (isThrowing) {
        // TODO: fmt::format introduced only in C++20
        std::string err2_msg = "!!!Unrecoverable!!! : " + std::string{cudaGetErrorString(err2)} +
                               " : detected by: " + func + " at: " + file + ":" + std::to_string(line) +
                               "\nThe error is reported there and may be caused by prior calls.\n";
        std::cerr << err2_msg << std::endl; // TODO: Logging
        throw IcicleError{err2, err2_msg};
      } else {
        err = err2;
      }
    }
    CHK_ERR(err, func, file, line);
  }

  return err;
}

// most common macros to use
#define CHK_INIT_IF_RETURN()                                                                                           \
  {                                                                                                                    \
    cudaError_t err_result = CHK_LAST();                                                                               \
    if (err_result != cudaSuccess) return err_result;                                                                  \
  }

#define CHK_IF_RETURN(val)                                                                                             \
  {                                                                                                                    \
    cudaError_t err_result = CHK_STICKY(val);                                                                          \
    if (err_result != cudaSuccess) return err_result;                                                                  \
  }

#endif
