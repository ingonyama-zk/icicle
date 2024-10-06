
# ICICLE Example: Arkworks and Icicle Scalar and Point Conversion

## Key Takeaway

This example demonstrates how to convert and manipulate field elements and elliptic curve points between Arkworks and ICICLE, specifically using the BN254 curve. It focuses on converting Arkworks types to ICICLE types and vice versa, ensuring seamless interaction between the two libraries.

> [!NOTE]
> Converting elements from other curves follows the same approach. Just replace the BN254 types in Arkworks and ICICLE with the corresponding types for your desired curve. The process remains the same.

> [!NOTE]
> Typically, elliptic curve points are converted once during setup and reused across multiple proofs, making conversion time less critical for these points. However, scalar values need to be converted per proof. EC point conversion is more expensive due to differences in memory layout, while scalar conversion mainly involves converting to/from Montgomery form. Future versions of ICICLE may natively support Montgomery representation for improved efficiency.

## Usage

```rust
// Generate a sequence of Arkworks scalars (e.g., [0, 1, 2, ..., size-1])
let ark_scalars = incremental_ark_scalars(size);
// Convert Arkworks scalars to ICICLE scalars
let icicle_scalars: DeviceVec<IcicleScalar> = ark_to_icicle_scalars(&ark_scalars);
// Alternatively, use transmute to reinterpret Arkworks memory as ICICLE, noting that it modifies in-place.
// Ensure you do not use the Arkworks scalars after this operation, as they are now ICICLE scalars.
let icicle_scalars: &mut [IcicleScalar] = transmute_ark_to_icicle_scalars(&mut ark_scalars);

// Generate Arkworks affine points (e.g., [g, 2*g, 3*g, ..., size*g])
let ark_affine_points = incremental_ark_affine_points(size);
// Convert Arkworks affine points to ICICLE affine points
let icicle_affine_points = ark_to_icicle_affine_points(&ark_affine_points);

// Generate Arkworks projective points (represented as Jacobian internally)
let ark_projective_points = incremental_ark_projective_points(size);
// Convert Arkworks projective points to ICICLE projective points
let icicle_projective_points = ark_to_icicle_projective_points(&ark_projective_points);
```

This example uses the `BN254` elliptic curve and showcases conversions between Arkworks and ICICLE for both affine and projective points. You can use this as a reference for other curves or use cases.

## What's in this example

1. Define the size of the dataset (scalars and points).
2. Set up the ICICLE backend (CPU or CUDA).
3. Generate incremental scalars and points using Arkworks.
4. Convert Arkworks field elements into ICICLE scalars.
5. Convert affine and projective points between Arkworks and Icicle.
6. Compute MSM (Multi-Scalar Multiplication) in both Arkworks and ICICLE, using either the CPU or CUDA backend.
7. Convert the MSM output back from ICICLE to Arkworks format and compare the results.

## Running the Example

To run the example, use the following commands based on your backend setup:

```sh
# For CPU
./run.sh -d CPU

# For CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

This will execute the example using either the CPU or CUDA backend, allowing you to test the conversion and computation process between Arkworks and ICICLE.
