#pragma once
#ifndef MSM_H
#define MSM_H

#include <cuda_runtime.h>

#include "curves/curve_config.cuh"
#include "primitives/affine.cuh"
#include "primitives/field.cuh"
#include "primitives/projective.cuh"
#include "utils/device_context.cuh"
#include "utils/error_handler.cuh"

/**
 * @namespace msm
 * Multi-scalar-multiplication, or MSM, is the sum of products of the form:
 * \f[
 *  MSM(s_i, P_i) = \sum_{i=1}^N s_i \cdot P_i
 * \f]
 * where \f$ \{P_i\} \f$ are elements of a certain group, \f$ \{s_i\} \f$ are scalars and \f$ N \f$ is the number of
 * terms. In cryptographic applications, prime-order subgroups of elliptic curve groups are typically used, so we refer
 * to group elements \f$ \{P_i\} \f$ as "points".
 *
 * To solve an MSM problem, we use an algorithm called the "bucket method". For a theoretical background on this
 * algorithm, see [this](https://www.youtube.com/watch?v=Bl5mQA7UL2I) great talk by Gus Gutoski.
 *
 * This codebase is based on and evolved from Matter Labs'
 * [Zprize
 * submission](https://github.com/matter-labs/z-prize-msm-gpu/blob/main/bellman-cuda-rust/bellman-cuda-sys/native/msm.cu).
 */
namespace msm {

  /**
   * @struct MSMConfig
   * Struct that encodes MSM parameters to be passed into the [MSM](@ref MSM) function. The intended use of this struct
   * is to create it using [DefaultMSMConfig](@ref DefaultMSMConfig) function and then you'll hopefully only need to
   * change a small number of default values for each of your MSMs.
   */
  struct MSMConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    int points_size;         /**< Number of points in the MSM. If a batch of MSMs needs to be computed, this should be
                              *   a number of different points. So, if each MSM re-uses the same set of points, this
                              *   variable is set equal to the MSM size. And if every MSM uses a distinct set of
                              *   points, it should be set to the product of MSM size and [batch_size](@ref
                              *   batch_size). Default value: 0 (meaning it's equal to the MSM size). */
    int precompute_factor;   /**< The number of extra points to pre-compute for each point. See the
                              *   [PrecomputeMSMBases](@ref PrecomputeMSMBases) function, `precompute_factor` passed
                              *   there needs to be equal to the one used here. Larger values decrease the
                              *   number of computations to make, on-line memory footprint, but increase the static
                              *   memory footprint. Default value: 1 (i.e. don't pre-compute). */
    int c;                   /**< \f$ c \f$ value, or "window bitsize" which is the main parameter of the "bucket
                              *   method" that we use to solve the MSM problem. As a rule of thumb, larger value
                              *   means more on-line memory footprint but also more parallelism and less computational
                              *   complexity (up to a certain point). Currently pre-computation is independent of
                              *   \f$ c \f$, however in the future value of \f$ c \f$ here and the one passed into the
                              *   [PrecomputeMSMBases](@ref PrecomputeMSMBases) function will need to be identical.
                              *    Default value: 0 (the optimal value of \f$ c \f$ is chosen automatically).  */
    int bitsize;             /**< Number of bits of the largest scalar. Typically equals the bitsize of scalar field,
                              *   but if a different (better) upper bound is known, it should be reflected in this
                              *   variable. Default value: 0 (set to the bitsize of scalar field). */
    int large_bucket_factor; /**< Variable that controls how sensitive the algorithm is to the buckets that occur
                              *   very frequently. Useful for efficient treatment of non-uniform distributions of
                              *   scalars and "top windows" with few bits. Can be set to 0 to disable separate
                              *   treatment of large buckets altogether. Default value: 10. */
    int batch_size;          /**< The number of MSMs to compute. Default value: 1. */
    bool are_scalars_on_device;       /**< True if scalars are on device and false if they're on host. Default value:
                                       *   false. */
    bool are_scalars_montgomery_form; /**< True if scalars are in Montgomery form and false otherwise. Default value:
                                       *   true. */
    bool are_points_on_device; /**< True if points are on device and false if they're on host. Default value: false. */
    bool are_points_montgomery_form; /**< True if coordinates of points are in Montgomery form and false otherwise.
                                      *   Default value: true. */
    bool are_results_on_device; /**< True if the results should be on device and false if they should be on host. If set
                                 *   to false, `is_async` won't take effect because a synchronization is needed to
                                 *   transfer results to the host. Default value: false. */
    bool is_big_triangle;       /**< Whether to do "bucket accumulation" serially. Decreases computational complexity
                                 *   but also greatly decreases parallelism, so only suitable for large batches of MSMs.
                                 *   Default value: false. */
    bool is_async;              /**< Whether to run the MSM asynchronously. If set to true, the MSM function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the MSM
                                 *   function will block the current CPU thread. */
  };

  /**
   * A function that returns the default value of [MSMConfig](@ref MSMConfig) for the [MSM](@ref MSM) function.
   * @return Default value of [MSMConfig](@ref MSMConfig).
   */
  template <typename A>
  MSMConfig DefaultMSMConfig();

  /**
   * A function that computes MSM: \f$ MSM(s_i, P_i) = \sum_{i=1}^N s_i \cdot P_i \f$.
   * @param scalars Scalars \f$ s_i \f$. In case of batch MSM, the scalars from all MSMs are concatenated.
   * @param points Points \f$ P_i \f$. In case of batch MSM, all *unique* points are concatenated.
   * So, if for example all MSMs share the same base points, they can be repeated only once.
   * @param msm_size MSM size \f$ N \f$. If a batch of MSMs (which all need to have the same size) is computed, this is
   * the size of 1 MSM.
   * @param config [MSMConfig](@ref MSMConfig) used in this MSM.
   * @param results Buffer for the result (or results in the case of batch MSM).
   * @tparam S Scalar field type.
   * @tparam A The type of points \f$ \{P_i\} \f$ which is typically an [affine
   * Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) point.
   * @tparam P Output type, which is typically a [projective
   * Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) point in our codebase.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   *
   */
  template <typename S, typename A, typename P>
  cudaError_t MSM(S* scalars, A* points, int msm_size, MSMConfig& config, P* results);

  /**
   * A function that precomputes MSM bases by extending them with their shifted copies.
   * e.g.:
   * Original points: \f$ P_0, P_1, P_2, ... P_{size} \f$
   * Extended points: \f$ P_0, P_1, P_2, ... P_{size}, 2^{l}P_0, 2^{l}P_1, ..., 2^{l}P_{size},
   * 2^{2l}P_0, 2^{2l}P_1, ..., 2^{2cl}P_{size}, ... \f$
   * @param bases Bases \f$ P_i \f$. In case of batch MSM, all *unique* points are concatenated.
   * @param bases_size Number of bases.
   * @param precompute_factor The number of total precomputed points for each base (including the base itself).
   * @param _c This is currently unused, but in the future precomputation will need to be aware of
   * the `c` value used in MSM (see [MSMConfig](@ref MSMConfig)). So to avoid breaking your code with this
   * upcoming change, make sure to use the same value of `c` in this function and in respective MSMConfig.
   * @param are_bases_on_device Whether the bases are on device.
   * @param ctx Device context specifying device id and stream to use.
   * @param output_bases Device-allocated buffer of size bases_size * precompute_factor for the extended bases.
   * @tparam A The type of points \f$ \{P_i\} \f$ which is typically an [affine
   * Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) point.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   *
   */
  template <typename A, typename P>
  cudaError_t PrecomputeMSMBases(
    A* bases,
    int bases_size,
    int precompute_factor,
    int _c,
    bool are_bases_on_device,
    device_context::DeviceContext& ctx,
    A* output_bases);

} // namespace msm


namespace msm {
  namespace {
    template <typename S> __global__ void find_cutoff_kernel(unsigned* v, unsigned size, unsigned cutoff, unsigned run_length, unsigned* result);

    template <typename P>
    __global__ void initialize_large_bucket_indices(
      unsigned* sorted_bucket_sizes_sum,
      unsigned nof_pts_per_thread,
      unsigned nof_large_buckets,
      // log_nof_buckets_to_compute should be equal to ceil(log(nof_buckets_to_compute))
      unsigned log_nof_large_buckets,
      unsigned* bucket_indices);

    template <typename E>
    __global__ void normalize_kernel(E* inout, E factor, int n);

    unsigned get_optimal_c(int bitsize);

    template <typename A, typename P>
    __global__ void left_shift_kernel(A* points, const unsigned shift, const unsigned count, A* points_out);

    template <typename P>
    __global__ void sum_reduction_variable_size_kernel(
      P* v,
      unsigned* bucket_sizes_sum,
      unsigned* bucket_sizes,
      unsigned* large_bucket_thread_indices,
      unsigned num_of_threads);

    template <typename P>
    __global__ void single_stage_multi_reduction_kernel(
      const P* v,
      P* v_r,
      unsigned block_size,
      unsigned write_stride,
      unsigned write_phase,
      unsigned step,
      unsigned num_of_threads);

    template <typename P>
    __global__ void initialize_buckets_kernel(P* buckets, unsigned N);

    template <typename S>
    __global__ void split_scalars_kernel(
      unsigned* buckets_indices,
      unsigned* point_indices,
      S* scalars,
      unsigned nof_scalars,
      unsigned points_size,
      unsigned msm_size,
      unsigned nof_bms,
      unsigned bm_bitsize,
      unsigned c,
      unsigned precomputed_bms_stride);

      template <typename P, typename S>
    __global__ void final_accumulation_kernel(
      const P* final_sums, P* final_results, unsigned nof_msms, unsigned nof_bms, unsigned nof_empty_bms, unsigned c);

      template <typename S, typename P, typename A>
    cudaError_t bucket_method_msm(
      unsigned bitsize,
      unsigned c,
      S* scalars,
      A* points,
      unsigned batch_size,      // number of MSMs to compute
      unsigned single_msm_size, // number of elements per MSM (a.k.a N)
      unsigned nof_points,      // number of EC points in 'points' array. Must be either (1) single_msm_size if MSMs are
                                // sharing points or (2) single_msm_size*batch_size otherwise
      P* final_result,
      bool are_scalars_on_device,
      bool are_scalars_montgomery_form,
      bool are_points_on_device,
      bool are_points_montgomery_form,
      bool are_results_on_device,
      bool is_big_triangle,
      int large_bucket_factor,
      int precompute_factor,
      bool is_async,
      cudaStream_t stream);

  }
  
}

#endif