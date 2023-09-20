#pragma once
#ifndef MSM_H
#define MSM_H

/**
 * @namespace msm
 * Multi-scalar-multiplication, or MSM, is the sum of products of the form:
 * \f[
 *  MSM(s_i, P_i) = \sum_{i=1}^N s_i \cdot P_i
 * \f]
 * where \f$ \{P_i\} \f$ are elements of a certain group, \f$ \{s_i\} \f$ are scalars and \f$ N \f$ is the number of terms.
 * In cryptographic applications, prime-order subgroups of elliptic curve groups are typically used, 
 * so we refer to group elements \f$ \{P_i\} \f$ as "points".
 *
 * To solve the MSM problem, we use an algorithm called the "bucket method". For a theoretical background on this algorithm, 
 * see [this](https://www.youtube.com/watch?v=Bl5mQA7UL2I) great talk by Gus Gutoski.
 */
namespace msm {

/**
 * @struct MSMConfig
 * Struct that encodes MSM parameters to be passed into the [msm_internal](@ref msm_internal) function.
 */
struct MSMConfig {
  bool scalars_on_device;         /**< True if scalars are on device and false if they're on host. Default value: false. */
  bool scalars_montgomery_form;   /**< True if scalars are in Montgomery form and false otherwise. Default value: true. */
  unsigned points_size;           /**< Number of points in the MSM. If a batch of MSMs needs to be computed, this should be a number 
                                   *   of different points. So, if each MSM re-uses the same set of points, this variable is set equal 
                                   *   to the MSM size. And if every MSM uses a distinct set of points, it should be set to the product of 
                                   *   MSM size and [batch_size](@ref batch_size). Default value: 0 (meaning it's equal to the MSM size). */
  unsigned precompute_factor;     /**< The number of extra points to pre-compute for each point. Larger values decrease the number of computations 
                                   *   to make, on-line memory footprint, but increase the static memory footprint. Default value: 1 (i.e. don't pre-compute). */
  bool points_on_device;          /**< True if points are on device and false if they're on host. Default value: false. */
  bool points_montgomery_form;    /**< True if coordinates of points are in Montgomery form and false otherwise. Default value: true. */
  unsigned batch_size;            /**< The number of MSMs to compute. Default value: 1. */
  bool result_on_device;          /**< True if the results should be on device and false if they should be on host. Default value: false. */
  unsigned c;                     /**< \f$ c \f$ value, or "window bitsize" which is the main parameter of the "bucket method"
                                   *   that we use to solve the MSM problem. As a rule of thumb, larger value means more on-line memory
                                   *   footprint but also more parallelism and less computational complexity (up to a certain point).
                                   *   Default value: 0 (the optimal value of \f$ c \f$ is chosen automatically). */
  unsigned bitsize;               /**< Number of bits of the largest scalar. Typically equals the bitsize of scalar field, but if a different 
                                   *   (better) upper bound is known, it should be reflected in this variable. Default value: 0 (set to the bitsize of scalar field). */
  bool big_triangle;              /**< Whether to do "bucket accumulation" serially. Decreases computational complexity, but also greatly 
                                   *   decreases parallelism, so only suitable for large batches of MSMs. Default value: false. */
  unsigned large_bucket_factor;   /**< Variable that controls how sensitive the algorithm is to the buckets that occur very frequently. 
                                   *   Useful for efficient treatment of non-uniform distributions of scalars and "top windows" with few bits.
                                   *   Can be set to 0 to disable separate treatment of large buckets altogether. Default value: 10. */
  unsigned device_id;             /**< Index of the GPU to run the MSM on. Default value: 0. */
  cudaStream_t stream;            /**< Stream to use. Default value: 0. */
};

/**
 * A function that computes MSM: \f$ MSM(s_i, P_i) = \sum_{i=1}^N s_i \cdot P_i \f$.
 * @param scalars Scalars \f$ s_i \f$. In case of batch MSM, the scalars from all MSMs are concatenated.
 * @param points Points \f$ P_i \f$. In case of batch MSM, all *unique* points are concatenated. So, if for example all MSMs share the same base points, they can be repeated only once.
 * @param msm_size MSM size \f$ N \f$. If a batch of MSMs (which all need to have the same size) is computed, this is the size of 1 MSM.
 * @param results Result (or results in the case of batch MSM).
 * @param config [MSMConfig](@ref MSMConfig) used in this MSM.
 * @tparam S Scalar field type.
 * @tparam A The type of points \f$ \{P_i\} \f$ which is typically an [affine Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) point.
 * @tparam P Output type, which is typically a [projective Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) point in our codebase.
 */
template <typename S, typename A, typename P>
void msm_internal(S* scalars, A* points, unsigned msm_size, MSMConfig config, P* results);

/**
 * A function that computes MSM by calling [msm_internal](@ref msm_internal) function with default [MSMConfig](@ref MSMConfig) values.
 */
template <typename S, typename A, typename P>
void msm(S* scalars, A* points, unsigned size, P* result);

} // namespace msm

#endif