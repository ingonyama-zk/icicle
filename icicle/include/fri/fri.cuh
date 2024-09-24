#pragma once
#ifndef FRI_H
#define FRI_H

#include <cuda_runtime.h>

#include "gpu-utils/device_context.cuh"

namespace fri {

  struct FriConfig {
    device_context::DeviceContext ctx;
    bool are_evals_on_device;
    bool are_domain_elements_on_device;
    bool are_results_on_device;
    bool is_async;
  };

  /**
   * @brief Folds a layer's evaluation into a degree d/2 evaluation using the provided folding factor alpha.
   *
   * @param evals Pointer to the array of evaluation in the current FRI layer.
   * @param domain_xs Pointer to a subset of line domain values.
   * @param alpha The folding factor used in the FRI protocol.
   * @param folded_evals Pointer to the array where the folded evaluations will be stored.
   * @param n The number of evaluations in the original layer (before folding).
   *
   * @tparam S The scalar field type used for domain_xs.
   * @tparam E The evaluation type, typically the same as the field element type.
   *
   * @note The size of the output array 'folded_evals' should be half of 'n', as folding reduces the number of
   * evaluations by half.
   */
  template <typename S, typename E>
  cudaError_t fold_line(E* eval, S* domain_xs, E alpha, E* folded_eval, int n, FriConfig& cfg);

  /**
   * @brief Folds a layer of FRI evaluations from a circle into a line.
   *
   * This function performs the folding operation in the FRI (Fast Reed-Solomon IOP of Proximity) protocol,
   * specifically for evaluations on a circle domain. It takes a layer of evaluations on a circle and folds
   * them into a line using the provided folding factor alpha.
   *
   * @param evals Pointer to the array of evaluations in the current FRI layer, representing points on a circle.
   * @param domain_ys Pointer to the array of y-coordinates of the circle points in the domain of the circle that evals
   * represents.
   * @param alpha The folding factor used in the FRI protocol.
   * @param folded_evals Pointer to the array where the folded evaluations (now on a line) will be stored.
   * @param n The number of evaluations in the original layer (before folding).
   *
   * @tparam S The scalar field type used for alpha and domain_ys.
   * @tparam E The evaluation type, typically the same as the field element type.
   *
   * @note The size of the output array 'folded_evals' should be half of 'n', as folding reduces the number of
   * evaluations by half.
   * @note This function is specifically designed for folding evaluations from a circular domain to a linear domain.
   */

  template <typename S, typename E>
  cudaError_t fold_circle_into_line(E* eval, S* domain_ys, E alpha, E* folded_eval, int n, FriConfig& cfg);

} // namespace fri

#endif