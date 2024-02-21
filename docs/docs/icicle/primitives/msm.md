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

$MSM(a, G) \in G$ - a single EC (elliptic curve point) point

In words, MSM is the sum of scalar and EC point multiplications. We can see from this definition that the core operations occurring are Modular Multiplication and Elliptic curve point addition. Each multiplication can be computed independently and then the products are summed, making MSM inherently parallelizable.

Accelerating MSM is crucial to a ZK protocol's performance due to the [large percent of run time](https://hackmd.io/@0xMonia/SkQ6-oRz3#Hardware-acceleration-in-action) they take when generating proofs.

You can learn more about how MSMs work from this [video](https://www.youtube.com/watch?v=Bl5mQA7UL2I) and from our resource list on [Ingopedia](https://www.ingonyama.com/ingopedia/msm).

# Using MSM

## Supported curves

MSM supports the following curves:

`bls12-377`, `bls12-381`, `bn-254`, `bw6-761`

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


### How do I toggle between the supported algorithms?

When creating your MSM Config you may state which algorithm you wish to use. `is_big_triangle=true` will activate Large triangle accumulation and `is_big_triangle=false` will activate Bucket accumulation.

```rust
...

let mut cfg_bls12377 = msm::get_default_msm_config::<BLS12377CurveCfg>();

// is_big_triangle will determine which algorithm to use 
cfg_bls12377.is_big_triangle = true;

msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
...
```

You may reference the rust code [here](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/wrappers/rust/icicle-core/src/msm/mod.rs#L54).

## MSM Modes

ICICLE MSM also supports two different modes `Batch MSM` and `Single MSM`

Batch MSM allows you to run many MSMs with a single API call, Single MSM will launch a single MSM computation.

### Which mode should I use?

This decision is highly dependent on your use case and design. However, if your design allows for it, using batch mode can significantly improve efficiency. Batch processing allows you to perform multiple MSMs leveraging the parallel processing capabilities of GPUs.

Single MSM mode should be used when batching isn't possible or when you have to run a single MSM.

### How do I toggle between MSM modes?

Toggling between MSM modes occurs automatically based on the number of results you are expecting from the `msm::msm` function. If you are expecting an array of `msm_results`, ICICLE will automatically split `scalars` and `points` into equal parts and run them as multiple MSMs in parallel.

```rust
...

let mut msm_result: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
msm::msm(&scalars, &points, &cfg, &mut msm_result).unwrap();

...
```

In the example above we allocate a single expected result which the MSM method will interpret as `batch_size=1` and run a single MSM.


In the next example, we are expecting 10 results which sets `batch_size=10` and runs 10 MSMs in batch mode.

```rust
...

let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(10).unwrap();
msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();

...
```

Here is a [reference](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/wrappers/rust/icicle-core/src/msm/mod.rs#L108) to the code which automatically sets the batch size. For more MSM examples have a look [here](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/examples/rust/msm/src/main.rs#L1).


## Support for G2 group

MSM also supports G2 group. 

Using MSM in G2 requires a G2 config, and of course your Points should also be G2 Points.

```rust
... 

let scalars = HostOrDeviceSlice::Host(upper_scalars[..size].to_vec());
let g2_points = HostOrDeviceSlice::Host(g2_upper_points[..size].to_vec());
let mut g2_msm_results: HostOrDeviceSlice<'_, G2Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
let mut g2_cfg = msm::get_default_msm_config::<G2CurveCfg>();

msm::msm(&scalars, &g2_points, &g2_cfg, &mut g2_msm_results).unwrap();

...
```

Here you can [find an example](https://github.com/ingonyama-zk/icicle/blob/5a96f9937d0a7176d88c766bd3ef2062b0c26c37/examples/rust/msm/src/main.rs#L114) of MSM on G2 Points.
