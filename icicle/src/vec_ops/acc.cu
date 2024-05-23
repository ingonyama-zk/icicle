#include <vector>
#include <cuda_runtime.h>

#include "vec_ops/acc.cuh"
namespace vec_ops {
  __global__ void accumulate_kernel(SecureColumn column, const SecureColumn other)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < column.packed_len) {
      float res_coeff = column.packed_at(i) + other.packed_at(i);
      column.set_packed(i, res_coeff);
    }
  }

  void accumulate(SecureColumn& column, const SecureColumn& other)
  {
    int packed_len = column.packed_len;

    // Allocate memory on the device
    SecureColumn d_column;
    SecureColumn d_other;

    cudaMalloc(&d_column.data, packed_len * sizeof(float));
    cudaMalloc(&d_other.data, packed_len * sizeof(float));
    d_column.packed_len = packed_len;
    d_other.packed_len = packed_len;

    // Copy data to the device
    cudaMemcpy(d_column.data, column.data, packed_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_other.data, other.data, packed_len * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (packed_len + threadsPerBlock - 1) / threadsPerBlock;
    accumulate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_column, d_other);

    // Copy the result back to the host
    cudaMemcpy(column.data, d_column.data, packed_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_column.data);
    cudaFree(d_other.data);
  }
} // namespace vec_ops