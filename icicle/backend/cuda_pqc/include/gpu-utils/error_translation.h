#pragma once

#include "icicle/errors.h"
#include "cuda_runtime.h"

static eIcicleError translateCudaError(cudaError_t cudaErr)
{
  switch (cudaErr) {
  case cudaSuccess:
    return eIcicleError::SUCCESS;
  case cudaErrorInvalidDevice:
    return eIcicleError::INVALID_DEVICE;
  case cudaErrorMemoryAllocation:
    return eIcicleError::OUT_OF_MEMORY;
  case cudaErrorInvalidDevicePointer:
  case cudaErrorInvalidHostPointer:
    return eIcicleError::INVALID_POINTER;
  case cudaErrorInitializationError:
  case cudaErrorInvalidResourceHandle:
    return eIcicleError::ALLOCATION_FAILED;
  case cudaErrorInvalidMemcpyDirection:
    return eIcicleError::COPY_FAILED;
  case cudaErrorSyncDepthExceeded:
  case cudaErrorLaunchTimeout:
  case cudaErrorLaunchIncompatibleTexturing:
  case cudaErrorLaunchFailure:
    return eIcicleError::SYNCHRONIZATION_FAILED;
  case cudaErrorInvalidValue:
    return eIcicleError::INVALID_ARGUMENT;
  default:
    return eIcicleError::UNKNOWN_ERROR;
  }
}
