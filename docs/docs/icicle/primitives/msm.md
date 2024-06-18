# MSM - Multi scalar multiplication

MSM stands for Multi scalar multiplication, its defined as:

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

In words, MSM is the sum of scalar and EC point multiplications. We can see from this definition that the core operations occurring are Modular Multiplication and Elliptic curve point addition. Its obvious that multiplication can be computed in parallel and then the products summed, making MSM inherently parallelizable.

Accelerating MSM is crucial to a ZK protocol's performance due to the [large percent of run time](https://hackmd.io/@0xMonia/SkQ6-oRz3#Hardware-acceleration-in-action) they take when generating proofs.

You can learn more about how MSMs work from this [video](https://www.youtube.com/watch?v=Bl5mQA7UL2I) and from our resource list on [Ingopedia](https://www.ingonyama.com/ingopedia/msm).

## Supported Bindings

- [Golang](../golang-bindings/msm.md)
- [Rust](../rust-bindings//msm.md)

## Algorithm description

We follow the bucket method algorithm. The GPU implementation consists of four phases:

1. Preparation phase - The scalars are split into smaller scalars of `c` bits each. These are the bucket indices. The points are grouped according to their corresponding bucket index and the buckets are sorted by size.
2. Accumulation phase - Each bucket accumulates all of its points using a single thread. More than one thread is assigned to large buckets, in proportion to their size. A bucket is considered large if its size is above the large bucket threshold that is determined by the `large_bucket_factor` parameter. The large bucket threshold is the expected avarage bucket size times the `large_bucket_factor` parameter.
3. Buckets Reduction phase - bucket results are multiplied by their corresponding bucket number and each bucket module is reduced to a small number of final results. By default this is done by and iterative algorithm which is highly parallel. Setting `is_big_triangle` to `true` will switch this phase to the running sum algorithm described in the above YouTube talk which is much less parallel.
4. Final accumulation phase - The final results from the last phase are accumulated using the double-and-add algorithm.

## MSM configuration



## Choosing optimal parameters

`is_big_triangle` should be `false` in almost all cases. It might provide better results only for very small MSMs (smaller than 2^8) with a large batch (larger than 100) but this should be tested per scenario.
Large buckets exist in two cases:
1. When the scalar distribution isn't uniform.
2. When `c` does not divide the scalar bit-size.
`large_bucket_factor` that is equal to 10 yields good results for most cases, but it's best to fine tune this parameter per `c` and per scalar distribution.
The two most important parameters for performance are `c` and the `precompute_factor`. They affect the number of EC additions as well as the memory size. When the points are not known in advance we cannot use precomputation. In this case the best `c` value is usually around log2(msm_size) - 4. However, in most protocols the points are known in advanced and precomputation can be used unless limited by memory. Usually it's best to use maximum precomputation (such that we end up with only a single bucket module) combined we a `c` value around log2(msm_size) - 1.

## Memory usage estimation

The main memory requirements of the MSM are the following:

Scalars - scalar_size * msm_size * batch_size
Scalar indices - unsigned_size * nof_bms * msm_size * batch_size * 6
Points - affine_size * msm_size * precomp_factor * batch_size
Buckets - projective_size * nof_bms * 2^c * batch_size

## Example parameters


### Bucket accumulation

The Bucket Accumulation algorithm is a method of dividing the overall MSM task into smaller, more manageable sub-tasks. It involves partitioning scalars and their corresponding points into different "buckets" based on the scalar values.

Bucket Accumulation can be more parallel-friendly because it involves dividing the computation into smaller, independent tasks, distributing scalar-point pairs into buckets and summing points within each bucket. This division makes it well suited for parallel processing on GPUs.

#### When should I use Bucket accumulation?

In scenarios involving large MSM computations with many scalar-point pairs, the ability to parallelize operations makes Bucket Accumulation more efficient. The larger the MSM task, the more significant the potential gains from parallelization.

### Large triangle accumulation

Large Triangle Accumulation is a method for optimizing MSM which focuses on reducing the number of point doublings in the computation. This algorithm is based on the observation that the number of point doublings can be minimized by structuring the computation in a specific manner.

#### When should I use Large triangle accumulation?

The Large Triangle Accumulation algorithm is more sequential in nature, as it builds upon each step sequentially (accumulating sums and then performing doubling). This structure can make it less suitable for parallelization but potentially more efficient for a **large batch of smaller MSM computations**.

## MSM Modes

ICICLE MSM also supports two different modes `Batch MSM` and `Single MSM`

Batch MSM allows you to run many MSMs with a single API call while single MSM will launch a single MSM computation.

### Which mode should I use?

This decision is highly dependent on your use case and design. However, if your design allows for it, using batch mode can significantly improve efficiency. Batch processing allows you to perform multiple MSMs simultaneously, leveraging the parallel processing capabilities of GPUs.

Single MSM mode should be used when batching isn't possible or when you have to run a single MSM.
