#pragma once

namespace icicle {

/**
 * @brief Enum representing various error codes for Icicle library operations.
 */
enum class IcicleError {
    SUCCESS = 0,                ///< Operation completed successfully
    INVALID_DEVICE,             ///< The specified device is invalid
    OUT_OF_MEMORY,              ///< Memory allocation failed due to insufficient memory
    INVALID_POINTER,            ///< The specified pointer is invalid
    ALLOCATION_FAILED,          ///< Memory allocation failed
    DEALLOCATION_FAILED,        ///< Memory deallocation failed
    COPY_FAILED,                ///< Data copy operation failed
    SYNCHRONIZATION_FAILED,     ///< Device synchronization failed
    STREAM_CREATION_FAILED,     ///< Stream creation failed
    STREAM_DESTRUCTION_FAILED,  ///< Stream destruction failed
    UNKNOWN_ERROR               ///< An unknown error occurred
};

/**
 * @brief Returns a human-readable string representation of an IcicleError.
 * 
 * @param error The IcicleError to get the string representation for.
 * @return const char* A string describing the error.
 */
const char* getErrorString(IcicleError error);

#define ICICLE_CHECK(api_call)                                    \
do {                                                              \
  using namespace icicle;                                         \
  IcicleError rv = (api_call);                                    \
  if (rv != IcicleError::SUCCESS) {                               \
    throw std::runtime_error("Icicle API failes with code " +     \
    std::string(getErrorString(rv)) +                             \
    " in " + __FILE__ + ":" + std::to_string(__LINE__));          \
  }                                                               \
} while(0) 


} // namespace icicle
