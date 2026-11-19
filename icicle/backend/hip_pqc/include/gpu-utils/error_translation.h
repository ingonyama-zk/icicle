#pragma once

#include "icicle/errors.h"
#include "cuda_to_hip.h"

// HIP-PQC port: translate hipError_t to the icicle frontend error enum. Mirrors
// the cuda_pqc translateCudaError, mapped onto the HIP error codes that exist
// in ROCm (HIP has no distinct InvalidHostPointer / SyncDepthExceeded codes).
static eIcicleError translateCudaError(hipError_t err)
{
  switch (err) {
  case hipSuccess:
    return eIcicleError::SUCCESS;
  case hipErrorInvalidDevice:
    return eIcicleError::INVALID_DEVICE;
  case hipErrorOutOfMemory:
    return eIcicleError::OUT_OF_MEMORY;
  case hipErrorInvalidDevicePointer:
    return eIcicleError::INVALID_POINTER;
  case hipErrorNotInitialized:
  case hipErrorInvalidResourceHandle:
    return eIcicleError::ALLOCATION_FAILED;
  case hipErrorInvalidMemcpyDirection:
    return eIcicleError::COPY_FAILED;
  case hipErrorLaunchTimeOut:
  case hipErrorLaunchFailure:
    return eIcicleError::SYNCHRONIZATION_FAILED;
  case hipErrorInvalidValue:
    return eIcicleError::INVALID_ARGUMENT;
  default:
    return eIcicleError::UNKNOWN_ERROR;
  }
}
