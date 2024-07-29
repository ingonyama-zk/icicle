# ICICLE example: MultiScalar Multiplication (MSM) in Rust

`ICICLE` provides Rust bindings to CUDA-accelerated C++ implementation of [Multi-Scalar Multiplication](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).

## Usage

```rust
msm(
  /* Scalars input vector */ &scalars, 
  /* Points input vector */ &points, 
  /* MSMConfig reference */ &cfg, 
  /* Projective point result */ &mut msm_results.as_slice()
).unwrap();
```
In this example we use `BN254` curve. The function computes $result = \sum_{i=0}^{size-1} scalars[i] \cdot points[i]$, where input `points[]` uses affine coordinates, and `result` uses projective coordinates.

## What's in the example

1. Define the size of MSM. 
2. Generate random inputs on-device
3. Configure MSM
4. Execute MSM on-device
5. Move the result on host

Running the example:
```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

