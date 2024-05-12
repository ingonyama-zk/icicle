#include <stdexcept>

#include "../../include/vec_ops/vec_ops.cuh"
#include "../../include/gpu-utils/device_context.cuh"
// #include "utils/mont.cuh"

namespace vec_ops {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E>
    void mul_kernel(const E* scalar_vec, const E* element_vec, int n, E* result)
    {
      return;
    }

    template <typename E, typename S>
    void mul_scalar_kernel(const E* element_vec, const S scalar, int n, E* result)
    {
      return;
    }

    template <typename E>
    void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      return;
    }

    template <typename E>
    void add_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      return;
    }

    template <typename E>
    void sub_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      return;
    }

    template <typename E>
    void transpose_kernel(const E* in, E* out, uint32_t row_size, uint32_t column_size)
    {
      return;
    }
  } // namespace

  typedef int cudaError_t;
  template <typename E, void (*Kernel)(const E*, const E*, int, E*)>
  cudaError_t vec_op(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return 0;
  }

  template <typename E>
  cudaError_t mul(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, mul_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t add(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, add_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t sub(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, sub_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t transpose_matrix(
    const E* mat_in,
    E* mat_out,
    uint32_t row_size,
    uint32_t column_size,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    return 0;
  }
} // namespace vec_ops