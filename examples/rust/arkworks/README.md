
# ICICLE Example: Arkworks and Icicle Scalar and Point Conversion

## Key Takeaway

This example demonstrates how to convert and manipulate field elements and elliptic curve points between Arkworks, for BN254 curve.

## Usage

```rust
let ark_scalars = incremental_ark_scalars(size); // generate scalars [0,1,2...size-1]
let icicle_scalars: DeviceVec<IcicleScalar> = ark_to_icicle_scalars(&ark_scalars);
// or transmute: reinterpret ark memory as ICICLE and use.
// Note that this function is converting from Montgomery in-place so make sure to not use the ark scalars
let icicle_scalars: &mut [IcicleScalar] = transmute_ark_to_icicle_scalars(&mut ark_scalars);

// Converting affine points (internally arkworks stores an additional byte so )
let ark_affine_points = incremental_ark_affine_points(size); // generate ark-affine points [g, 2*g, 3*g...size*g]
let icicle_affine_points = ark_to_icicle_affine_points(&ark_affine_points);

// Converting projective points (internally in arkworks it is Jacobian)
let ark_projective_points = incremental_ark_projective_points(size); // generate ark-projective points [g, 2*g, 3*g...size*g]
let icicle_projective_points = ark_to_icicle_projective_points(&ark_projective_points);
```

In this example, we use the `BN254` elliptic curve to handle field elements and elliptic curve points in both affine and projective forms.
This example should be used as reference for other curves and usecases.

## What's in this example

1. Define the size of the dataset (scalars and points).
2. Set up the ICICLE backend (CPU or CUDA).
3. Generate incremental scalars and points in Arkworks.
4. Copy Arkworks field elements to Icicle scalars.
5. Convert affine and projective points between Arkworks and Icicle.
6. [Future] Execute operations on-device (CPU or CUDA) and move results back to the host.
7. [Future] Compare results between Arkworks and Icicle.

## Running the Example

```sh
# For CPU
./run.sh -d CPU

# For CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```
