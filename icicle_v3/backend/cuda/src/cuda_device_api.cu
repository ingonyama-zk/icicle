#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

#include "cuda_runtime.h"

using namespace icicle;

class CudaDeviceAPI : public DeviceAPI
{
public:
  eIcicleError set_device(const Device& device) override
  {
    cudaError_t err = cudaSetDevice(device.id);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  // Memory management
  eIcicleError allocate_memory(void** ptr, size_t size) const override
  {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError allocate_memory_async(void** ptr, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMallocAsync(ptr, size, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
  }

  eIcicleError free_memory(void* ptr) const override
  {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError free_memory_async(void* ptr, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaFreeAsync(ptr, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::DEALLOCATION_FAILED;
  }

  eIcicleError get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/) const override
  {
    cudaError_t err = cudaMemGetInfo(&free, &total);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
  }

  // Data transfer
  eIcicleError copy_to_host(void* dst, const void* src, size_t size) const override
  {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copy_to_host_async(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copy_to_device(void* dst, const void* src, size_t size) const override
  {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
  }

  eIcicleError copy_to_device_async(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
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
  eIcicleError create_stream(icicleStreamHandle* stream) const override
  {
    cudaStream_t cudaStream;
    cudaError_t err = cudaStreamCreate(&cudaStream);
    *stream = reinterpret_cast<icicleStreamHandle>(cudaStream);
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_CREATION_FAILED;
  }

  eIcicleError destroy_stream(icicleStreamHandle stream) const override
  {
    cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }

  eIcicleError get_device_properties(DeviceProperties& properties) const override
  {
    properties.using_host_memory = false;
    properties.num_memory_regions = 1;
    properties.supports_pinned_memory = false; // TODO support it for compatible devices
    return eIcicleError::SUCCESS;
  }
};

REGISTER_DEVICE_API("CUDA", CudaDeviceAPI);