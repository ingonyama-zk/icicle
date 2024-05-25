#include <vector>
#include <cuda_runtime.h>
namespace vec_ops {
  template <typename T>

  struct SecureColumn {
    T* data;
    int packed_len;

    __device__ float packed_at(int index) const { return data[index]; }

    __device__ void set_packed(int index, float value) { data[index] = value; }
  };

  /**
   * A function that adds two vectors element-wise.
   * @param column First input vector.
   * @param other Second input vector.
   * @param stream The CUDA stream.
   */

  template <typename T>
  void accumulate_async(SecureColumn<T>& column, const SecureColumn<T>& other, cudaStream_t stream = (cudaStream_t)0);

} // namespace vec_ops