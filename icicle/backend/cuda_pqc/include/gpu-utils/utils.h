#pragma once

#include "cuda.h"
#include "gpu-utils/error_handler.h"

template <typename E>
static E* allocate_on_device(size_t byte_size, cudaStream_t cuda_stream, bool is_async = true)
{
  E* device_mem = nullptr;
  if (is_async) {
    cudaMallocAsync(&device_mem, byte_size, cuda_stream);
  } else {
    cudaMalloc(&device_mem, byte_size);
  }
  CHK_LAST();

  return device_mem;
}

template <typename E>
static const E*
allocate_and_copy_to_device(const E* host_mem, size_t byte_size, cudaStream_t cuda_stream, bool is_async = true)
{
  E* device_mem = nullptr;
  if (is_async) {
    cudaMallocAsync(&device_mem, byte_size, cuda_stream);
    cudaMemcpyAsync(device_mem, host_mem, byte_size, cudaMemcpyHostToDevice, cuda_stream);
  } else {
    cudaMalloc(&device_mem, byte_size);
    cudaMemcpy(device_mem, host_mem, byte_size, cudaMemcpyHostToDevice);
  }
  CHK_LAST();

  return device_mem;
}

template <typename E>
static const E* copy_to_device(
  const E* host_mem, size_t byte_size, bool are_inputs_on_device, cudaStream_t cuda_stream, bool is_async = true)
{
  E* device_mem = nullptr;
  auto memKind = are_inputs_on_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
  if (is_async) {
    cudaMemcpyAsync(device_mem, host_mem, byte_size, memKind, cuda_stream);
  } else {
    cudaMemcpy(device_mem, host_mem, byte_size, memKind);
  }

  CHK_LAST();

  return device_mem;
}