# MSM

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
pub struct MSMConfig {
    pub stream_handle: IcicleStreamHandle,    
    pub precompute_factor: i32,
    pub c: i32,
    pub bitsize: i32,    
    batch_size: i32,
    are_bases_shared: bool,
    are_scalars_on_device: bool,
    pub are_scalars_montgomery_form: bool,
    are_points_on_device: bool,
    pub are_points_montgomery_form: bool,
    are_results_on_device: bool,    
    pub is_async: bool,
    pub ext: ConfigExtension,
}
```

- **`stream_handle: IcicleStreamHandle`**: Specifies a stream for asynchronous execution.
- **`precompute_factor: i32`**: Determines the number of extra points to pre-compute for each point, affecting memory footprint and performance.
- **`c: i32`**: The "window bitsize," a parameter controlling the computational complexity and memory footprint of the MSM operation.
- **`bitsize: i32`**: The number of bits of the largest scalar, typically equal to the bit size of the scalar field.
- **`batch_size: i32`**: The number of MSMs to compute in a single batch, for leveraging parallelism.
- **`are_scalars_montgomery_form`**: Set to `true` if scalars are in montgomery form.
- **`are_points_montgomery_form`**: Set to `true` if points are in montgomery form.
- **`are_scalars_on_device: bool`**, **`are_points_on_device: bool`**, **`are_results_on_device: bool`**: Indicate whether the corresponding buffers are on the device memory.
- **`is_async: bool`**: Whether to perform the MSM operation asynchronously.
- **`ext: ConfigExtension`**: extended configuration for backend.

### Usage

The `msm` function is designed to compute the sum of multiple scalar-point multiplications efficiently. It supports both single MSM operations and batched operations for increased performance. The configuration allows for detailed control over the execution environment and performance characteristics of the MSM operation.

When performing MSM operations, it's crucial to match the size of the `scalars` and `points` arrays correctly and ensure that the `results` buffer is appropriately sized to hold the output. The `MSMConfig` should be set up to reflect the specifics of the operation, including whether the operation should be asynchronous and any device-specific settings.

## Example

```rust
// Using bls12-377 curve
use icicle_bls12_377::curve::{CurveCfg, G1Projective, ScalarCfg};
use icicle_core::{curve::Curve, msm, msm::MSMConfig, traits::GenerateRandom};
use icicle_runtime::{device::Device, memory::HostSlice};

fn main() {
    // Load backend and set device ...

    // Randomize inputs
    let size = 1024;
    let points = CurveCfg::generate_random_affine_points(size);
    let scalars = ScalarCfg::generate_random(size);

    let mut msm_results = vec![G1Projective::zero(); 1];
    msm::msm(
        HostSlice::from_slice(&scalars),
        HostSlice::from_slice(&points),
        &MSMConfig::default(),
        HostSlice::from_mut_slice(&mut msm_results[..]),
    )
    .unwrap();
    println!("MSM result = {:?}", msm_results);
}

```

## Batched msm

For batch msm, simply allocate the results array with size corresponding to batch size and set the `are_bases_shared` flag in config struct.

## Precomputationg

Precomputes bases for the multi-scalar multiplication (MSM) by extending each base point with its multiples, facilitating more efficient MSM calculations.

```rust
/// Returns `Ok(())` if no errors occurred or a `eIcicleError` otherwise.
pub fn precompute_bases<C: Curve + MSM<C>>(
    points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
    config: &MSMConfig,
    output_bases: &mut DeviceSlice<Affine<C>>,
) -> Result<(), eIcicleError>;
```

### Parameters

- **`points`**: The original set of affine points (\(P_1, P_2, ..., P_n\)) to be used in the MSM. For batch MSM operations, this should include all unique points concatenated together.
- **`msm_size`**: The size of a single msm in order to determine optimal parameters.
- **`cfg`**: The MSM configuration parameters.
- **`output_bases`**: The output buffer for the extended bases. Its size must be `points.len() * precompute_factor`. This buffer should be allocated on the device for GPU computations.

#### Returns

`Ok(())` if the operation is successful, or an `eIcicleError` error otherwise.

## Parameters for optimal performance

Please refer to the [primitive description](../primitives/msm#choosing-optimal-parameters)
