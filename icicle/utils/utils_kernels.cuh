#pragma once
#ifndef UTILS_KERNELS_H
#define UTILS_KERNELS_H

namespace utils_internal {

/**
 * Multiply the elements of an input array by a scalar in-place. Used for normalization in iNTT.
 * @param arr input array.
 * @param n size of arr.
 * @param n_inv scalar of type S (scalar).
 */
template <typename E, typename S>
__global__ void template_normalize_kernel(E* arr, S scalar, int n);

template <typename E, typename S>
__global__ void batchVectorMult(E* element_vec, S* scalar_vec, int n_scalars, int batch_size);

} // namespace utils_internal

#endif
