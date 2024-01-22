#pragma once
#ifndef LDE_H
#define LDE_H

#include "device_context.cuh"

/**
 * @namespace vec_ops
 * This namespace contains methods for performing element-wise arithmetic operations on vectors.
 */
namespace vec_ops {

  /**
   * A function that multiplies two vectors element-wise.
   * @param vec_a First input vector.
   * @param vec_b Second input vector.
   * @param n Size of vectors `vec_a` and `vec_b`.
   * @param is_on_device If true, inputs and outputs are on device, if false - on the host.
   * @param is_montgomery If true, inputs are expected to be in Montgomery form and results are returned in Montgomery
   * form. If false - inputs and outputs are non-Montgomery.
   * @param ctx [DeviceContext](@ref device_context::DeviceContext) used in this method.
   * @param result Resulting vector - element-wise product of `vec_a` and `vec_b`, can be the same pointer as `vec_b`.
   * @tparam S The type of scalars `vec_a`.
   * @tparam E The type of elements `vec_b` and `result`. Often (but not always) `E=S`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E, typename S>
  cudaError_t
  Mul(S* vec_a, E* vec_b, int n, bool is_on_device, bool is_montgomery, device_context::DeviceContext ctx, E* result);

  /**
   * A function that adds two vectors element-wise.
   * @param vec_a First input vector.
   * @param vec_b Second input vector.
   * @param n Size of vectors `vec_a` and `vec_b`.
   * @param is_on_device If true, inputs and outputs are on device, if false - on the host.
   * @param ctx [DeviceContext](@ref device_context::DeviceContext) used in this method.
   * @param result Resulting vector - element-wise sum of `vec_a` and `vec_b`, can be the same pointer as `vec_a` or
   * `vec_b`. If inputs are in Montgomery form, the result is too, and vice versa: non-Montgomery inputs produce
   * non-Montgomery result.
   * @tparam E The type of elements `vec_a`, `vec_b` and `result`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E>
  cudaError_t Add(E* vec_a, E* vec_b, int n, bool is_on_device, device_context::DeviceContext ctx, E* result);

  /**
   * A function that subtracts two vectors element-wise.
   * @param vec_a First input vector.
   * @param vec_b Second input vector.
   * @param n Size of vectors `vec_a` and `vec_b`.
   * @param is_on_device If true, inputs and outputs are on device, if false - on the host.
   * @param ctx [DeviceContext](@ref device_context::DeviceContext) used in this method.
   * @param result Resulting vector - element-wise difference of `vec_a` and `vec_b`, can be the same pointer as `vec_a`
   * or `vec_b`. If inputs are in Montgomery form, the result is too, and vice versa: non-Montgomery inputs produce
   * non-Montgomery result.
   * @tparam E The type of elements `vec_a`, `vec_b` and `result`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E>
  cudaError_t Sub(E* vec_a, E* vec_b, int n, bool is_on_device, device_context::DeviceContext ctx, E* result);

} // namespace vec_ops

#endif
