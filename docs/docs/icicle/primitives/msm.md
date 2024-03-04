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

## Supported curves

MSM supports the following curves:

`bls12-377`, `bls12-381`, `bn254`, `bw6-761`, `grumpkin`


## Supported Bindings

- [Golang](../golang-bindings/msm.md)
- [Rust](../rust-bindings//msm.md)

## Supported algorithms

Our MSM implementation supports two algorithms `Bucket accumulation` and `Large triangle accumulation`.

### Bucket accumulation

The Bucket Accumulation algorithm is a method of dividing the overall MSM task into smaller, more manageable sub-tasks. It involves partitioning scalars and their corresponding points into different "buckets" based on the scalar values.

Bucket Accumulation can be more parallel-friendly because it involves dividing the computation into smaller, independent tasks, distributing scalar-point pairs into buckets and summing points within each bucket. This division makes it well suited for parallel processing on GPUs.

#### When should I use Bucket accumulation?

In scenarios involving large MSM computations with many scalar-point pairs, the ability to parallelize operations makes Bucket Accumulation more efficient. The larger the MSM task, the more significant the potential gains from parallelization.

### Large triangle accumulation

Large Triangle Accumulation is a method for optimizing MSM which focuses on reducing the number of point doublings in the computation. This algorithm is based on the observation that the number of point doublings can be minimized by structuring the computation in a specific manner.

#### When should I use Large triangle accumulation?

The Large Triangle Accumulation algorithm is more sequential in nature, as it builds upon each step sequentially (accumulating sums and then performing doubling). This structure can make it less suitable for parallelization but potentially more efficient for a <b>large batch of smaller MSM computations</b>.

## MSM Modes

ICICLE MSM also supports two different modes `Batch MSM` and `Single MSM`

Batch MSM allows you to run many MSMs with a single API call, Single MSM will launch a single MSM computation.

### Which mode should I use?

This decision is highly dependent on your use case and design. However, if your design allows for it, using batch mode can significantly improve efficiency. Batch processing allows you to perform multiple MSMs leveraging the parallel processing capabilities of GPUs.

Single MSM mode should be used when batching isn't possible or when you have to run a single MSM.
