#include "icicle/errors.h"

namespace icicle {

/**
 * @brief Returns a human-readable string representation of an IcicleError.
 * 
 * @param error The IcicleError to get the string representation for.
 * @return const char* A string describing the error.
 */
const char* getErrorString(IcicleError error) {
    switch (error) {
        case IcicleError::SUCCESS:
            return "Success";
        case IcicleError::INVALID_DEVICE:
            return "Invalid device";
        case IcicleError::OUT_OF_MEMORY:
            return "Out of memory";
        case IcicleError::INVALID_POINTER:
            return "Invalid pointer";
        case IcicleError::ALLOCATION_FAILED:
            return "Memory allocation failed";
        case IcicleError::DEALLOCATION_FAILED:
            return "Memory deallocation failed";
        case IcicleError::COPY_FAILED:
            return "Data copy failed";
        case IcicleError::SYNCHRONIZATION_FAILED:
            return "Synchronization failed";
        case IcicleError::STREAM_CREATION_FAILED:
            return "Stream creation failed";
        case IcicleError::STREAM_DESTRUCTION_FAILED:
            return "Stream destruction failed";
        case IcicleError::UNKNOWN_ERROR:
        default:
            return "Unknown error";
    }
}

} // namespace icicle
