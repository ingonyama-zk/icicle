#pragma once
#ifndef LDE_H
#define LDE_H

#include "gpu-utils/device_context.cuh"


#include <cublas_v2.h>
#define TILE_DIM   16
#define BLOCK_ROWS 8 // 32

/**
 * @namespace vec_ops
 * This namespace contains methods for performing element-wise arithmetic operations on vectors.
 */
namespace vec_ops {

  /**
   * @struct VecOpsConfig
   * Struct that encodes NTT parameters to be passed into the [NTT](@ref NTT) function.
   */
  struct VecOpsConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream. */

    bool is_a_on_device; /**< True if `a` is on device and false if it is not. Default value: false. */

    bool is_b_on_device; /**< True if `b` is on device and false if it is not. Default value: false. */

    bool is_result_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */

    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */
  };

  /**
   * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
   * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
   */
  static VecOpsConfig
  DefaultVecOpsConfig(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    VecOpsConfig config = {
      ctx,   // ctx
      false, // is_a_on_device
      false, // is_b_on_device
      false, // is_result_on_device
      false, // is_async
    };
    return config;
  }

  /**
   * A function that multiplies two vectors element-wise.
   * @param vec_a First input vector.
   * @param vec_b Second input vector.
   * @param n Size of vectors `vec_a` and `vec_b`.
   * @param config Configuration of the operation.
   * @param result Resulting vector - element-wise product of `vec_a` and `vec_b`, can be the same pointer as `vec_b`.
   * @tparam S The type of scalars `vec_a`.
   * @tparam E The type of elements `vec_b` and `result`. Often (but not always) `E=S`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E, typename S>
  cudaError_t Mul(const S* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result);

  /**
   * A function that adds two vectors element-wise.
   * @param vec_a First input vector.
   * @param vec_b Second input vector.
   * @param n Size of vectors `vec_a` and `vec_b`.
   * @param config Configuration of the operation.
   * @param result Resulting vector - element-wise sum of `vec_a` and `vec_b`, can be the same pointer as `vec_a` or
   * `vec_b`. If inputs are in Montgomery form, the result is too, and vice versa: non-Montgomery inputs produce
   * non-Montgomery result.
   * @tparam E The type of elements `vec_a`, `vec_b` and `result`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E>
  cudaError_t Add(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result);

  /**
   * A function that subtracts two vectors element-wise.
   * @param vec_a First input vector.
   * @param vec_b Second input vector.
   * @param n Size of vectors `vec_a` and `vec_b`.
   * @param config Configuration of the operation.
   * @param result Resulting vector - element-wise difference of `vec_a` and `vec_b`, can be the same pointer as `vec_a`
   * or `vec_b`. If inputs are in Montgomery form, the result is too, and vice versa: non-Montgomery inputs produce
   * non-Montgomery result.
   * @tparam E The type of elements `vec_a`, `vec_b` and `result`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E>
  cudaError_t Sub(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result);

  /**
   * Transposes an input matrix out-of-place inside GPU.
   * for example: for ([a[0],a[1],a[2],a[3]], 2, 2) it returns
   * [a[0],a[2],a[1],a[3]].
   * @param mat_in array of some object of type E of size row_size * column_size.
   * @param arr_out buffer of the same size as `mat_in` to write the transpose matrix into.
   * @param row_size size of rows.
   * @param column_size size of columns.
   * @param ctx Device context.
   * @param on_device Whether the input and output are on device.
   * @param is_async Whether to run the vector operations asynchronously.
   * @tparam E The type of elements `mat_in' and `mat_out`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E>
  cudaError_t transpose_matrix(
    const E* mat_in,
    E* mat_out,
    uint32_t row_size,
    uint32_t column_size,
    const device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async);

  struct BitReverseConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream. */
    bool is_input_on_device;  /**< True if `input` is on device and false if it is not. Default value: false. */
    bool is_output_on_device; /**< True if `output` is on device and false if it is not. Default value: false. */
    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */
  };
  static BitReverseConfig
  DefaultBitReverseConfig(const device_context::DeviceContext& ctx = device_context::get_default_device_context())
  {
    BitReverseConfig config = {
      ctx,   // ctx
      false, // is_input_on_device
      false, // is_output_on_device
      false, // is_async
    };
    return config;
  }

  template <typename T>
  __global__ void transpose_inplace(T* data)
  {
    __shared__ T tile_s[TILE_DIM][TILE_DIM + 1];
    __shared__ T tile_d[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
      int dx = blockIdx.y * TILE_DIM + threadIdx.x;
      int dy = blockIdx.x * TILE_DIM + threadIdx.y;
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile_s[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile_d[threadIdx.y + j][threadIdx.x] = data[(dy + j) * width + dx];
      __syncthreads();
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        data[(dy + j) * width + dx] = tile_s[threadIdx.x][threadIdx.y + j];
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        data[(y + j) * width + x] = tile_d[threadIdx.x][threadIdx.y + j];
    }

    else if (blockIdx.y == blockIdx.x) { // handle on-diagonal case
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile_s[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];
      __syncthreads();
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        data[(y + j) * width + x] = tile_s[threadIdx.x][threadIdx.y + j];
    }
  }
} // namespace vec_ops

#endif
