#pragma once

#include <stdexcept>
#include <iostream>
#include <sstream>

namespace icicle {

  /**
   * @brief Enum representing various error codes for Icicle library operations.
   */
  enum class eIcicleError {
    SUCCESS = 0,               ///< Operation completed successfully
    INVALID_DEVICE,            ///< The specified device is invalid
    OUT_OF_MEMORY,             ///< Memory allocation failed due to insufficient memory
    INVALID_POINTER,           ///< The specified pointer is invalid
    ALLOCATION_FAILED,         ///< Memory allocation failed
    DEALLOCATION_FAILED,       ///< Memory deallocation failed
    COPY_FAILED,               ///< Data copy operation failed
    SYNCHRONIZATION_FAILED,    ///< Device synchronization failed
    STREAM_CREATION_FAILED,    ///< Stream creation failed
    STREAM_DESTRUCTION_FAILED, ///< Stream destruction failed
    API_NOT_IMPLEMENTED,       ///< The API is not implemented for a device
    INVALID_ARGUMENT,          ///< Invalid argument passed
    UNKNOWN_ERROR              ///< An unknown error occurred
  };

  /**
   * @brief Returns a human-readable string representation of an eIcicleError.
   *
   * @param error The eIcicleError to get the string representation for.
   * @return const char* A string describing the error.
   */
  static const char* get_error_string(eIcicleError error)
  {
    switch (error) {
    case eIcicleError::SUCCESS:
      return "eIcicleError::SUCCESS";
    case eIcicleError::INVALID_DEVICE:
      return "eIcicleError::INVALID_DEVICE";
    case eIcicleError::OUT_OF_MEMORY:
      return "eIcicleError::OUT_OF_MEMORY";
    case eIcicleError::INVALID_POINTER:
      return "eIcicleError::INVALID_POINTER";
    case eIcicleError::ALLOCATION_FAILED:
      return "eIcicleError::ALLOCATION_FAILED";
    case eIcicleError::DEALLOCATION_FAILED:
      return "eIcicleError::DEALLOCATION_FAILED";
    case eIcicleError::COPY_FAILED:
      return "eIcicleError::COPY_FAILED";
    case eIcicleError::SYNCHRONIZATION_FAILED:
      return "eIcicleError::SYNCHRONIZATION_FAILED";
    case eIcicleError::STREAM_CREATION_FAILED:
      return "eIcicleError::STREAM_CREATION_FAILED";
    case eIcicleError::STREAM_DESTRUCTION_FAILED:
      return "eIcicleError::STREAM_DESTRUCTION_FAILED";
    case eIcicleError::API_NOT_IMPLEMENTED:
      return "eIcicleError::API_NOT_IMPLEMENTED";
    case eIcicleError::INVALID_ARGUMENT:
      return "eIcicleError::INVALID_ARGUMENT";
    case eIcicleError::UNKNOWN_ERROR:
    default:
      return "eIcicleError::UNKNOWN_ERROR";
    }
  }

#define ICICLE_CHECK(api_call)                                                                                         \
  do {                                                                                                                 \
    using namespace icicle;                                                                                            \
    eIcicleError rv = (api_call);                                                                                      \
    if (rv != eIcicleError::SUCCESS) {                                                                                 \
      throw std::runtime_error(                                                                                        \
        "Icicle API fails with code " + std::string(get_error_string(rv)) + " in " + __FILE__ + ":" +                  \
        std::to_string(__LINE__));                                                                                     \
    }                                                                                                                  \
  } while (0)

  void inline throw_icicle_error(
    eIcicleError err, const char* const reason, const char* const func, const char* const file, const int line)
  {
    std::string err_msg = std::string{get_error_string(err)} + " : by: " + func + " at: " + file + ":" +
                          std::to_string(line) + " error: " + reason;
    std::cerr << err_msg << std::endl; // TODO: Logging
    throw std::runtime_error(err_msg);
  }

#define THROW_ICICLE_ERR(val, reason) throw_icicle_error(val, reason, __FUNCTION__, __FILE__, __LINE__)

  class AssertHelper
  {
  public:
    AssertHelper(const char* condition, const char* function, const char* file, int line) : os()
    {
      os << "Assertion failed: (" << condition << "), function " << function << ", file " << file << ", line " << line
         << ". ";
    }

    template <typename T>
    AssertHelper& operator<<(const T& msg)
    {
      os << msg;
      return *this;
    }

    ~AssertHelper() noexcept(false) { throw std::runtime_error(os.str()); }

  private:
    std::ostringstream os;
  };

#define ICICLE_ASSERT(condition)                                                                                       \
  if (!(condition)) AssertHelper(#condition, __FUNCTION__, __FILE__, __LINE__)

} // namespace icicle
