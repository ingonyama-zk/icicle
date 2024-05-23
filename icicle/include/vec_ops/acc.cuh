#include <vector>
#include <cuda_runtime.h>
namespace vec_ops {
  struct SecureColumn {
    float* data;
    int packed_len;

    __device__ float packed_at(int index) const { return data[index]; }

    __device__ void set_packed(int index, float value) { data[index] = value; }
  };

  /**
   * A function that adds two vectors element-wise.
   * @param column First input vector.
   * @param other Second input vector.
   */
  void accumulate(SecureColumn& column, const SecureColumn& other);
} // namespace vec_ops