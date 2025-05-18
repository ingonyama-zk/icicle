# ECNTT

## ECNTT Method

The `ecntt` function computes the Elliptic Curve Number Theoretic Transform (EC-NTT) or its inverse on a batch of points of a curve.

```rust
pub fn ecntt<C: Curve>(
    input: &(impl HostOrDeviceSlice<Projective<C>> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<C::ScalarField>,
    output: &mut (impl HostOrDeviceSlice<Projective<C>> + ?Sized),
) -> Result<(), eIcicleError>
```

## Parameters

- **`input`**: The input data as a slice of `Projective<C>`. This represents points on a specific elliptic curve `C`.
- **`dir`**: The direction of the NTT. It can be `NTTDir::kForward` for forward NTT or `NTTDir::kInverse` for inverse NTT.
- **`cfg`**: The NTT configuration object of type `NTTConfig<C::ScalarField>`. This object specifies parameters for the NTT computation, such as the batch size and algorithm to use.
- **`output`**: The output buffer to write the results into. This should be a slice of `Projective<C>` with the same size as the input.

## Return Value

- **`Result<(), eIcicleError>`**: This function returns an `eIcicleError` which is a wrapper type that indicates success or failure of the NTT computation. On success, it contains `Ok(())`.
