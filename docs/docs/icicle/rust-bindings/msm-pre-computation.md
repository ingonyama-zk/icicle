# MSM Pre computation

### `precompute_bases`

Precomputes bases for the multi-scalar multiplication (MSM) by extending each base point with its multiples, facilitating more efficient MSM calculations.

```rust
pub fn precompute_bases<C: Curve + MSM<C>>(
    points: &HostOrDeviceSlice<Affine<C>>,
    precompute_factor: i32,
    _c: i32,
    ctx: &DeviceContext,
    output_bases: &mut HostOrDeviceSlice<Affine<C>>,
) -> IcicleResult<()>
```


#### Parameters

- **`points`**: The original set of affine points (\(P_1, P_2, ..., P_n\)) to be used in the MSM. For batch MSM operations, this should include all unique points concatenated together.
- **`precompute_factor`**: Specifies the total number of points to precompute for each base, including the base point itself. This parameter directly influences the memory requirements and the potential speedup of the MSM operation.
- **`_c`**: Currently unused. Intended for future use to align with the `c` parameter in `MSMConfig`, ensuring the precomputation is compatible with the bucket method's window size used in MSM.
- **`ctx`**: The device context specifying the device ID and stream for execution. This context determines where the precomputation is performed (e.g., on a specific GPU).
- **`output_bases`**: The output buffer for the extended bases. Its size must be `points.len() * precompute_factor`. This buffer should be allocated on the device for GPU computations.

#### Returns

`Ok(())` if the operation is successful, or an `IcicleResult` error otherwise.

#### Description

This function extends each provided base point \(P\) with its multiples \(2^lP, 2^{2l}P, ..., 2^{(precompute_factor - 1) \cdot l}P\), where \(l\) is a level of precomputation determined by the `precompute_factor`. The extended set of points facilitates faster MSM computations by allowing the MSM algorithm to leverage precomputed multiples of base points, reducing the number of point additions required during the computation.

The precomputation process is crucial for optimizing MSM operations, especially when dealing with large sets of points and scalars. By precomputing and storing multiples of the base points, the MSM function can more efficiently compute the scalar-point multiplications.

#### Example Usage

```rust
let device_context = DeviceContext::default_for_device(0); // Use the default device
let precompute_factor = 4; // Number of points to precompute
let mut extended_bases = HostOrDeviceSlice::cuda_malloc(expected_size).expect("Failed to allocate memory for extended bases");

// Precompute the bases using the specified factor
precompute_bases(&points, precompute_factor, 0, &device_context, &mut extended_bases)
    .expect("Failed to precompute bases");
```

### Benchmarks

Benchmarks where preformed on a Nvidia RTX 3090Ti.

| Pre-computation factor | bn254 size `2^20` MSM, ms.  | bn254 size `2^12` MSM, size `2^10` batch, ms. | bls12-381 size `2^20` MSM, ms. | bls12-381 size `2^12` MSM, size `2^10` batch, ms. |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | 14.1  | 82.8  | 25.5  | 136.7  |
| 2  | 11.8  | 76.6  | 20.3  | 123.8  |
| 4  | 10.9  | 73.8  | 18.1  | 117.8  |
| 8  | 10.6  | 73.7  | 17.2  | 116.0  |
