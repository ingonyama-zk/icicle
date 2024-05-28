#include "errors.h"

namespace icicle {

  /**
   * @brief Returns a human-readable string representation of an eIcicleError.
   *
   * @param error The eIcicleError to get the string representation for.
   * @return const char* A string describing the error.
   */
  const char* get_error_string(eIcicleError error)
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

} // namespace icicle
