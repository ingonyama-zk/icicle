#pragma once
#ifndef LDE_H
#define LDE_H

/**
 * @namespace lde
 * LDE (stands for low degree extension) contains [NTT](@ref ntt)-based methods for translating between coefficient and evaluation domains of polynomials.
 * It also contains methods for element-wise manipulation of vectors, which is useful for working with polynomials in evaluation domain.
 */
namespace lde {

/**
 * A function that multiplies two vectors element-wise.
 * @param vec_a First input vector.
 * @param vec_b Second input vector.
 * @param n Size of vectors `vec_a` and `vec_b`.
 * @param is_on_device If true, inputs and outputs are on device, if false - on the host.
 * @param is_montgomery If true, inputs are expected to be Montgomery form and results are retured in Montgomery form.
 * If false - inputs and outputs are non-Montgomery.
 * @param ctx [DeviceContext](@ref device_context::DeviceContext) used in this method.
 * @param result Resulting vector - element-wise product of `vec_a` and `vec_b`, can be the same pointer as `vec_b`.
 * @tparam S The type of scalars `vec_a`.
 * @tparam E The type of elements `vec_b` and `result`. Often (but not always), `E=S`.
 * @return `cudaSuccess` if the execution was successful and an error code otherwise.
 */
template <typename E, typename S>
cudaError_t mul(S* vec_a, E* vec_b, size_t n, bool is_on_device, bool is_montgomery, device_context::DeviceContext ctx, E* result);

/**
 * A function that adds two vectors element-wise.
 * @param vec_a First input vector.
 * @param vec_b Second input vector.
 * @param n Size of vectors `vec_a` and `vec_b`.
 * @param is_on_device If true, inputs and outputs are on device, if false - on the host.
 * @param ctx [DeviceContext](@ref device_context::DeviceContext) used in this method.
 * @param result Resulting vector - element-wise sum of `vec_a` and `vec_b`, can be the same pointer as `vec_a` or `vec_b`.
 * @tparam E The type of elements `vec_a`, `vec_b` and `result`.
 * @return `cudaSuccess` if the execution was successful and an error code otherwise.
 */
template <typename E>
cudaError_t add(E* vec_a, E* vec_b, size_t n, bool is_on_device, device_context::DeviceContext ctx, E* result);

/**
 * A function that subtracts two vectors element-wise.
 * @param vec_a First input vector.
 * @param vec_b Second input vector.
 * @param n Size of vectors `vec_a` and `vec_b`.
 * @param is_on_device If true, inputs and outputs are on device, if false - on the host.
 * @param ctx [DeviceContext](@ref device_context::DeviceContext) used in this method.
 * @param result Resulting vector - element-wise difference of `vec_a` and `vec_b`, can be the same pointer as `vec_a` or `vec_b`.
 * @tparam E The type of elements `vec_a`, `vec_b` and `result`.
 * @return `cudaSuccess` if the execution was successful and an error code otherwise.
 */
template <typename E>
cudaError_t sub(E* vec_a, E* vec_b, size_t n, bool is_on_device, device_context::DeviceContext ctx, E* result);

} // namespace lde

#endif
