#include "errors.h"

namespace icicle {

  /**
   * @brief Returns a human-readable string representation of an IcicleError.
   *
   * @param error The IcicleError to get the string representation for.
   * @return const char* A string describing the error.
   */
  const char* getErrorString(IcicleError error)
  {
    switch (error) {
    case IcicleError::SUCCESS:
      return "IcicleError::SUCCESS";
    case IcicleError::INVALID_DEVICE:
      return "IcicleError::INVALID_DEVICE";
    case IcicleError::OUT_OF_MEMORY:
      return "IcicleError::OUT_OF_MEMORY";
    case IcicleError::INVALID_POINTER:
      return "IcicleError::INVALID_POINTER";
    case IcicleError::ALLOCATION_FAILED:
      return "IcicleError::ALLOCATION_FAILED";
    case IcicleError::DEALLOCATION_FAILED:
      return "IcicleError::DEALLOCATION_FAILED";
    case IcicleError::COPY_FAILED:
      return "IcicleError::COPY_FAILED";
    case IcicleError::SYNCHRONIZATION_FAILED:
      return "IcicleError::SYNCHRONIZATION_FAILED";
    case IcicleError::STREAM_CREATION_FAILED:
      return "IcicleError::STREAM_CREATION_FAILED";
    case IcicleError::STREAM_DESTRUCTION_FAILED:
      return "IcicleError::STREAM_DESTRUCTION_FAILED";
    case IcicleError::API_NOT_IMPLEMENTED:
      return "IcicleError::API_NOT_IMPLEMENTED";
    case IcicleError::INVALID_ARGUMENT:
      return "IcicleError::INVALID_ARGUMENT";
    case IcicleError::UNKNOWN_ERROR:
    default:
      return "IcicleError::UNKNOWN_ERROR";
    }
  }

} // namespace icicle
