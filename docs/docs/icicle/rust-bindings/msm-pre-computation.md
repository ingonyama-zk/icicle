# MSM Pre computation

To understand the theory behind MSM pre computation technique refer to Niall Emmart's [talk](https://youtu.be/KAWlySN7Hm8?feature=shared&t=1734).

## `precompute_points`

Precomputes bases for the multi-scalar multiplication (MSM) by extending each base point with its multiples, facilitating more efficient MSM calculations.

```rust
pub fn precompute_points<C: Curve + MSM<C>>(
    points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
    msm_size: i32,
    cfg: &MSMConfig,
    output_bases: &mut DeviceSlice<Affine<C>>,
) -> IcicleResult<()>
```

### Parameters

- **`points`**: The original set of affine points (\(P_1, P_2, ..., P_n\)) to be used in the MSM. For batch MSM operations, this should include all unique points concatenated together.
- **`msm_size`**: The size of a single msm in order to determine optimal parameters.
- **`cfg`**: The MSM configuration parameters.
- **`output_bases`**: The output buffer for the extended bases. Its size must be `points.len() * precompute_factor`. This buffer should be allocated on the device for GPU computations.

#### Returns

`Ok(())` if the operation is successful, or an `IcicleResult` error otherwise.

#### Description

This function extends each provided base point $(P)$ with its multiples $(2^lP, 2^{2l}P, ..., 2^{(precompute_factor - 1) \cdot l}P)$, where $(l)$ is a level of precomputation determined by the `precompute_factor`. The extended set of points facilitates faster MSM computations by allowing the MSM algorithm to leverage precomputed multiples of base points, reducing the number of point additions required during the computation.

The precomputation process is crucial for optimizing MSM operations, especially when dealing with large sets of points and scalars. By precomputing and storing multiples of the base points, the MSM function can more efficiently compute the scalar-point multiplications.

#### Example Usage

```rust
let cfg = MSMConfig::default();
let precompute_factor = 4; // Number of points to precompute
let mut extended_bases = HostOrDeviceSlice::cuda_malloc(expected_size).expect("Failed to allocate memory for extended bases");

// Precompute the bases using the specified factor
precompute_points(&points, msm_size, &cfg, &mut extended_bases)
    .expect("Failed to precompute bases");
```
