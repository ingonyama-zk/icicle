# MSM

## Example

```rust
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarCfg};
use icicle_core::{curve::Curve, msm, traits::GenerateRandom};
use icicle_cuda_runtime::{memory::HostOrDeviceSlice, stream::CudaStream};

fn main() {
    let size: usize = 1 << 10; // Define the number of points and scalars

    // Generate random points and scalars
    println!("Generating random G1 points and scalars for BN254...");
    let points = CurveCfg::generate_random_affine_points(size);
    let scalars = ScalarCfg::generate_random(size);

    // Wrap points and scalars in HostOrDeviceSlice for MSM
    let points_host = HostOrDeviceSlice::Host(points);
    let scalars_host = HostOrDeviceSlice::Host(scalars);

    // Allocate memory on the CUDA device for MSM results
    let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).expect("Failed to allocate CUDA memory for MSM results");

    // Create a CUDA stream for asynchronous execution
    let stream = CudaStream::create().expect("Failed to create CUDA stream");
    let mut cfg = msm::MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true; // Enable asynchronous execution

    // Execute MSM on the device
    println!("Executing MSM on device...");
    msm::msm(&scalars_host, &points_host, &cfg, &mut msm_results).expect("Failed to execute MSM");

    // Synchronize CUDA stream to ensure MSM execution is complete
    stream.synchronize().expect("Failed to synchronize CUDA stream");

    // Optionally, move results to host for further processing or printing
    println!("MSM execution complete.");
}
```

## MSM API Overview

```rust
pub fn msm<C: Curve>(
    scalars: &HostOrDeviceSlice<C::ScalarField>,
    points: &HostOrDeviceSlice<Affine<C>>,
    cfg: &MSMConfig,
    results: &mut HostOrDeviceSlice<Projective<C>>,
) -> IcicleResult<()>
```

### Parameters

- **`scalars`**: A buffer containing the scalar values to be multiplied with corresponding points.
- **`points`**: A buffer containing the points to be multiplied by the scalars.
- **`cfg`**: MSM configuration specifying additional parameters for the operation.
- **`results`**: A buffer where the results of the MSM operations will be stored.

### MSM Config

```rust
pub struct MSMConfig<'a> {
    pub ctx: DeviceContext<'a>,
    points_size: i32,
    pub precompute_factor: i32,
    pub c: i32,
    pub bitsize: i32,
    pub large_bucket_factor: i32,
    batch_size: i32,
    are_scalars_on_device: bool,
    pub are_scalars_montgomery_form: bool,
    are_points_on_device: bool,
    pub are_points_montgomery_form: bool,
    are_results_on_device: bool,
    pub is_big_triangle: bool,
    pub is_async: bool,
}
```

- **`ctx: DeviceContext`**: Specifies the device context, device id and the CUDA stream for asynchronous execution.
- **`point_size: i32`**:
- **`precompute_factor: i32`**: Determines the number of extra points to pre-compute for each point, affecting memory footprint and performance.
- **`c: i32`**: The "window bitsize," a parameter controlling the computational complexity and memory footprint of the MSM operation.
- **`bitsize: i32`**: The number of bits of the largest scalar, typically equal to the bit size of the scalar field.
- **`large_bucket_factor: i32`**: Adjusts the algorithm's sensitivity to frequently occurring buckets, useful for non-uniform scalar distributions.
- **`batch_size: i32`**: The number of MSMs to compute in a single batch, for leveraging parallelism.
- **`are_scalars_montgomery_form`**: Set to `true` if scalars are in montgomery form.
- **`are_points_montgomery_form`**: Set to `true` if points are in montgomery form.
- **`are_scalars_on_device: bool`**, **`are_points_on_device: bool`**, **`are_results_on_device: bool`**: Indicate whether the corresponding buffers are on the device memory.
- **`is_big_triangle`**: If `true` MSM will run in Large triangle accumulation if `false` Bucket accumulation will be chosen. Default value: false.
- **`is_async: bool`**: Whether to perform the MSM operation asynchronously.

### Usage

The `msm` function is designed to compute the sum of multiple scalar-point multiplications efficiently. It supports both single MSM operations and batched operations for increased performance. The configuration allows for detailed control over the execution environment and performance characteristics of the MSM operation.

When performing MSM operations, it's crucial to match the size of the `scalars` and `points` arrays correctly and ensure that the `results` buffer is appropriately sized to hold the output. The `MSMConfig` should be set up to reflect the specifics of the operation, including whether the operation should be asynchronous and any device-specific settings.

## How do I toggle between the supported algorithms?

When creating your MSM Config you may state which algorithm you wish to use. `is_big_triangle=true` will activate Large triangle reduction and `is_big_triangle=false` will activate iterative reduction.

```rust
...

let mut cfg_bls12377 = msm::get_default_msm_config::<BLS12377CurveCfg>();

// is_big_triangle will determine which algorithm to use 
cfg_bls12377.is_big_triangle = true;

msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
...
```

You may reference the rust code [here](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/wrappers/rust/icicle-core/src/msm/mod.rs#L54).

## How do I toggle between MSM modes?

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

## Parameters for optimal performance

Please refer to the [primitive description](../primitives/msm.md#choosing-optimal-parameters)

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
