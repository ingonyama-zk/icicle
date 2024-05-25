#include <vector>
#include <cuda_runtime.h>

#include "vec_ops/acc.cuh"
#include <stdio.h>
#include <cassert>
#include <gpu-utils/error_handler.cuh>
namespace vec_ops {

  template <typename T>
  __global__ void accumulate_kernel(T* column, const T* other, int length)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) { column[i] += other[i]; }
  }

  template <typename T>
  void accumulate_async(SecureColumn<T>& column, const SecureColumn<T>& other, cudaStream_t stream)
  {
    int packed_len = column.packed_len;

    // Allocate memory on the device
    SecureColumn<T> d_column;
    SecureColumn<T> d_other;

    cudaMallocAsync(&d_column.data, packed_len * sizeof(T), stream);
    cudaMallocAsync(&d_other.data, packed_len * sizeof(T), stream);
    d_column.packed_len = packed_len;
    d_other.packed_len = packed_len;

    // Copy data to the device
    cudaMemcpyAsync(d_column.data, column.data, packed_len * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_other.data, other.data, packed_len * sizeof(T), cudaMemcpyHostToDevice, stream);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (packed_len + threadsPerBlock - 1) / threadsPerBlock;
    accumulate_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_column.data, d_other.data, d_column.packed_len);

    // Copy the result back to the host
    cudaMemcpyAsync(column.data, d_column.data, packed_len * sizeof(T), cudaMemcpyDeviceToHost, stream);

    // Free device memory
    cudaFreeAsync(d_column.data, stream);
    cudaFreeAsync(d_other.data, stream);
  }
} // namespace vec_ops