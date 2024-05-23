#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

#include "cuda_runtime.h"

using namespace icicle;

class CUDADeviceAPI : public DeviceAPI
{
public:
  eIcicleError setDevice(const Device& device) const
  {
    cudaError_t err = cudaSetDevice(device.id);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  // Memory management
  eIcicleError allocateMemory(const Device& device, void** ptr, size_t size) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError allocateMemoryAsync(const Device& device, void** ptr, size_t size, IcicleStreamHandle stream) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMallocAsync(ptr, size, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError freeMemory(const Device& device, void* ptr) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError freeMemoryAsync(const Device& device, void* ptr, IcicleStreamHandle stream) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaFreeAsync(ptr, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError getAvailableMemory(const Device& device, size_t& total /*OUT*/, size_t& free /*OUT*/) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
  }

  // Data transfer
  eIcicleError copyToHost(const Device& device, void* dst, const void* src, size_t size) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError
  copyToHostAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copyToDevice(const Device& device, void* dst, const void* src, size_t size) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError
  copyToDeviceAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  // Synchronization
  eIcicleError synchronize(const Device& device, IcicleStreamHandle stream = nullptr) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err =
      (stream == nullptr) ? cudaDeviceSynchronize() : cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::SYNCHRONIZATION_FAILED;
  }

  // Stream management
  eIcicleError createStream(const Device& device, IcicleStreamHandle* stream) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaStream_t cudaStream;
    cudaError_t err = cudaStreamCreate(&cudaStream);
    *stream = reinterpret_cast<IcicleStreamHandle>(cudaStream);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_CREATION_FAILED;
  }

  eIcicleError destroyStream(const Device& device, IcicleStreamHandle stream) override
  {
    if (eIcicleError err = setDevice(device); err != eIcicleError::SUCCESS) return err;
    cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }
};

REGISTER_DEVICE_API("CUDA", CUDADeviceAPI);