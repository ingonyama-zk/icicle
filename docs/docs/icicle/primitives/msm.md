# MSM - Multi scalar multiplication

MSM stands for Multi scalar multiplication, it's defined as:

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>M</mi>
  <mi>S</mi>
  <mi>M</mi>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>,</mo>
  <mi>G</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP" movablelimits="false">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>j</mi>
      <mo>=</mo>
      <mn>0</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>n</mi>
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </munderover>
  <msub>
    <mi>a</mi>
    <mi>j</mi>
  </msub>
  <msub>
    <mi>G</mi>
    <mi>j</mi>
  </msub>
</math>

Where

$G_j \in G$ - points from an Elliptic Curve group.

$a_0, \ldots, a_n$ - Scalars

$MSM(a, G) \in G$ - a single EC (elliptic curve) point

In words, MSM is the sum of scalar and EC point multiplications. We can see from this definition that the core operations occurring are Modular Multiplication and Elliptic curve point addition. It's obvious that multiplication can be computed in parallel and then the products are summed, making MSM inherently parallelizable.

Accelerating MSM is crucial to a ZK protocol's performance due to the [large percent of run time](https://hackmd.io/@0xMonia/SkQ6-oRz3#Hardware-acceleration-in-action) they take when generating proofs.

You can learn more about how MSMs work from this [video](https://www.youtube.com/watch?v=Bl5mQA7UL2I) and from our resource list on [Ingopedia](https://www.ingonyama.com/ingopedia/msm).

## Supported Bindings

- [Golang](../golang-bindings/msm.md)
- [Rust](../rust-bindings//msm.md)

## Algorithm description

We follow the bucket method algorithm. The GPU implementation consists of four phases:

1. Preparation phase - The scalars are split into smaller scalars of `c` bits each. These are the bucket indices. The points are grouped according to their corresponding bucket index and the buckets are sorted by size.
2. Accumulation phase - Each bucket accumulates all of its points using a single thread. More than one thread is assigned to large buckets, in proportion to their size. A bucket is considered large if its size is above the large bucket threshold that is determined by the `large_bucket_factor` parameter. The large bucket threshold is the expected average bucket size times the `large_bucket_factor` parameter.
3. Buckets Reduction phase - bucket results are multiplied by their corresponding bucket number and each bucket module is reduced to a small number of final results. By default, this is done by an iterative algorithm which is highly parallel. Setting `is_big_triangle` to `true` will switch this phase to the running sum algorithm described in the above YouTube talk which is much less parallel.
4. Final accumulation phase - The final results from the last phase are accumulated using the double-and-add algorithm.

## Batched MSM

The MSM supports batch mode - running multiple MSMs in parallel. It's always better to use the batch mode instead of running single msms in serial as long as there is enough memory available. We support running a batch of MSMs that share the same points as well as a batch of MSMs that use different points.

## MSM configuration

```cpp
  /**
   * @struct MSMConfig
   * Struct that encodes MSM parameters to be passed into the [MSM](@ref MSM) function. The intended use of this struct
   * is to create it using [default_msm_config](@ref default_msm_config) function and then you'll hopefully only need to
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
                              *   [precompute_msm_points](@ref precompute_msm_points) function, `precompute_factor` passed
                              *   there needs to be equal to the one used here. Larger values decrease the
                              *   number of computations to make, on-line memory footprint, but increase the static
                              *   memory footprint. Default value: 1 (i.e. don't pre-compute). */
    int c;                   /**< \f$ c \f$ value, or "window bitsize" which is the main parameter of the "bucket
                              *   method" that we use to solve the MSM problem. As a rule of thumb, larger value
                              *   means more on-line memory footprint but also more parallelism and less computational
                              *   complexity (up to a certain point). Currently pre-computation is independent of
                              *   \f$ c \f$, however in the future value of \f$ c \f$ here and the one passed into the
                              *   [precompute_msm_points](@ref precompute_msm_points) function will need to be identical.
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
```

## Choosing optimal parameters

`is_big_triangle` should be `false` in almost all cases. It might provide better results only for very small MSMs (smaller than 2^8^) with a large batch (larger than 100) but this should be tested per scenario.
Large buckets exist in two cases:
1. When the scalar distribution isn't uniform.
2. When `c` does not divide the scalar bit-size.

`large_bucket_factor` that is equal to 10 yields good results for most cases, but it's best to fine tune this parameter per `c` and per scalar distribution.
The two most important parameters for performance are `c` and the `precompute_factor`. They affect the number of EC additions as well as the memory size. When the points are not known in advance we cannot use precomputation. In this case the best `c` value is usually around $log_2(msmSize) - 4$. However, in most protocols the points are known in advance and precomputation can be used unless limited by memory. Usually it's best to use maximum precomputation (such that we end up with only a single bucket module) combined with a `c` value around $log_2(msmSize) - 1$.

## Memory usage estimation

The main memory requirements of the MSM are the following:

- Scalars - `sizeof(scalar_t) * msm_size * batch_size`
- Scalar indices - `~6 * sizeof(unsigned) * nof_bucket_modules * msm_size * batch_size`
- Points - `sizeof(affine_t) * msm_size * precomp_factor * batch_size`
- Buckets - `sizeof(projective_t) * nof_bucket_modules * 2^c * batch_size`

where `nof_bucket_modules =  ceil(ceil(bitsize / c) / precompute_factor)`

During the MSM computation first the memory for scalars and scalar indices is allocated, then the indices are freed and points and buckets are allocated. This is why a good estimation for the required memory is the following formula:

$max(scalars + scalarIndices, scalars + points + buckets)$

This gives a good approximation within 10% of the actual required memory for most cases.

## Example parameters

Here is a useful table showing optimal parameters for different MSMs. They are optimal for BLS12-377 curve when running on NVIDIA GeForce RTX 3090 Ti. This is the configuration used:

```cpp
  msm::MSMConfig config = {
    ctx,            // DeviceContext
    N,              // points_size
    precomp_factor, // precompute_factor
    user_c,         // c
    0,              // bitsize
    10,             // large_bucket_factor
    batch_size,     // batch_size
    false,          // are_scalars_on_device
    false,          // are_scalars_montgomery_form
    true,           // are_points_on_device
    false,          // are_points_montgomery_form
    true,           // are_results_on_device
    false,          // is_big_triangle
    true            // is_async
  };
```

Here are the parameters and the results for the different cases:

| MSM size | Batch size | Precompute factor | c | Memory estimation (GB) | Actual memory (GB) | Single MSM time (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 1 | 1 | 9 | 0.00227 | 0.00277 | 9.2 |
| 10 | 1 | 23 | 11 | 0.00259 | 0.00272 | 1.76 |
| 10 | 1000 | 1 | 7 | 0.94 | 1.09 | 0.051 |
| 10 | 1000 | 23 | 11 | 2.59 | 2.74 | 0.025 |
| 15 | 1 | 1 | 11 | 0.011 | 0.019 | 9.9 |
| 15 | 1 | 16 | 16 | 0.061 | 0.065 | 2.4 |
| 15 | 100 | 1 | 11 | 1.91 | 1.92 | 0.84 |
| 15 | 100 | 19 | 14 | 6.32 | 6.61 | 0.56 |
| 18 | 1 | 1 | 14 | 0.128 | 0.128 | 14.4 |
| 18 | 1 | 15 | 17 | 0.40 | 0.42 | 5.9 |
| 22 | 1 | 1 | 17 | 1.64 | 1.65 | 68 |
| 22 | 1 | 13 | 21 | 5.67 | 5.94 | 54 |
| 24 | 1 | 1 | 18 | 6.58 | 6.61 | 232 |
| 24 | 1 | 7 | 21 | 12.4 | 13.4 | 199 |

The optimal values can vary per GPU and per curve. It is best to try a few combinations until you get the best results for your specific case.
