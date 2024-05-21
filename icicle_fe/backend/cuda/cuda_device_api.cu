#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

#include "cuda_runtime.h"

using namespace icicle;

class CUDADeviceAPI : public DeviceAPI
{
public:
    // Memory management
    IcicleError allocateMemory(const Device& device, void** ptr, size_t size) override
    {
        cudaError_t err = cudaMalloc(ptr, size);
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::ALLOCATION_FAILED;
    }

    IcicleError allocateMemoryAsync(const Device& device, void** ptr, size_t size, IcicleStreamHandle stream) override
    {
        cudaError_t err = cudaMallocAsync(ptr, size, reinterpret_cast<cudaStream_t>(stream));
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::ALLOCATION_FAILED;
    }

    IcicleError freeMemory(const Device& device, void* ptr) override
    {
        cudaError_t err = cudaFree(ptr);
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::DEALLOCATION_FAILED;
    }

    IcicleError freeMemoryAsync(const Device& device, void* ptr, IcicleStreamHandle stream) override
    {
        cudaError_t err = cudaFreeAsync(ptr, reinterpret_cast<cudaStream_t>(stream));
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::DEALLOCATION_FAILED;
    }

    IcicleError getAvailableMemory(const Device& device, size_t& total /*OUT*/, size_t& free /*OUT*/) override
    {
        size_t freeMem = 0, totalMem = 0;
        cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err != cudaSuccess) {
            return IcicleError::UNKNOWN_ERROR;
        }

        free = freeMem;
        total = totalMem;
        return IcicleError::SUCCESS;
    }

    // Data transfer
    IcicleError copyToHost(const Device& device, void* dst, const void* src, size_t size) override
    {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::COPY_FAILED;
    }

    IcicleError copyToHostAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) override
    {
        cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream));
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::COPY_FAILED;
    }

    IcicleError copyToDevice(const Device& device, void* dst, const void* src, size_t size) override
    {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::COPY_FAILED;
    }

    IcicleError copyToDeviceAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) override
    {
        cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream));
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::COPY_FAILED;
    }

    // Synchronization
    IcicleError synchronize(const Device& device, IcicleStreamHandle stream = nullptr) override
    {
        cudaError_t err = (stream == nullptr) ? cudaDeviceSynchronize() : cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::SYNCHRONIZATION_FAILED;
    }

    // Stream management
    IcicleError createStream(const Device& device, IcicleStreamHandle* stream) override
    {
        cudaStream_t cudaStream;
        cudaError_t err = cudaStreamCreate(&cudaStream);
        *stream = reinterpret_cast<IcicleStreamHandle>(cudaStream);
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::STREAM_CREATION_FAILED;
    }

    IcicleError destroyStream(const Device& device, IcicleStreamHandle stream) override
    {
        cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
        return (err == cudaSuccess) ? IcicleError::SUCCESS : IcicleError::STREAM_DESTRUCTION_FAILED;
    }
};

REGISTER_DEVICE_API("CUDA", CUDADeviceAPI);