#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

#include "cuda_runtime.h"

using namespace icicle;

class CUDADeviceAPI : public DeviceAPI
{
public:
  eIcicleError setDevice(const Device& device) override
  {
    cudaError_t err = cudaSetDevice(device.id);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  // Memory management
  eIcicleError allocateMemory(void** ptr, size_t size) const override
  {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError allocateMemoryAsync(void** ptr, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMallocAsync(ptr, size, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError freeMemory(void* ptr) const override
  {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError freeMemoryAsync(void* ptr, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaFreeAsync(ptr, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError getAvailableMemory(size_t& total /*OUT*/, size_t& free /*OUT*/) const override
  {
    cudaError_t err = cudaMemGetInfo(&free, &total);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
  }

  // Data transfer
  eIcicleError copyToHost(void* dst, const void* src, size_t size) const override
  {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copyToHostAsync(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copyToDevice(void* dst, const void* src, size_t size) const override
  {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copyToDeviceAsync(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  // Synchronization
  eIcicleError synchronize(icicleStreamHandle stream = nullptr) const override
  {
    cudaError_t err =
      (stream == nullptr) ? cudaDeviceSynchronize() : cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::SYNCHRONIZATION_FAILED;
  }

  // Stream management
  eIcicleError createStream(icicleStreamHandle* stream) const override
  {
    cudaStream_t cudaStream;
    cudaError_t err = cudaStreamCreate(&cudaStream);
    *stream = reinterpret_cast<icicleStreamHandle>(cudaStream);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_CREATION_FAILED;
  }

  eIcicleError destroyStream(icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }
};

REGISTER_DEVICE_API("CUDA", CUDADeviceAPI);