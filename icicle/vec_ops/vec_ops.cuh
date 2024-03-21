#pragma once
#ifndef LDE_H
#define LDE_H

#include "gpu-utils/device_context.cuh"

/**
 * @namespace vec_ops
 * This namespace contains methods for performing element-wise arithmetic operations on vectors.
 */
namespace vec_ops {

  /**
   * @struct VecOpsConfig
   * Struct that encodes NTT parameters to be passed into the [NTT](@ref NTT) function.
   */
  template <typename S>
  struct VecOpsConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream. */

    bool is_a_on_device; /**< True if `a` is on device and false if it is not. Default value: false. */

    bool is_b_on_device; /**< True if `b` is on device and false if it is not. Default value: false. */

    bool is_result_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */

    bool is_result_montgomery_form; /**< True if `result` vector should be in Montgomery form and false otherwise.
                                     *   Default value: false. */

    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */
  };

  /**
   * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
   * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
   */
  template <typename S>
  VecOpsConfig<S> DefaultVecOpsConfig()
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    VecOpsConfig<S> config = {
      ctx,   // ctx
      false, // is_a_on_device
      false, // is_b_on_device
      false, // is_result_on_device
      false, // is_result_montgomery_form
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
  cudaError_t Mul(S* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result);

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
  cudaError_t Add(E* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result);

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
  cudaError_t Sub(E* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result);
} // namespace vec_ops

#endif
