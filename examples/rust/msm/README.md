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
./run.sh CPU # to use CPU backend
./run.sh CUDA # to load and use CUDA backend
```

> [!NOTE]
> The default sizes are 2^10 - 2^10. You can change this by passing the `--lower_bound_log_size <size> --upper_bound_log_size <size>` options. To change the size range to 2^21 - 2^24, run the example like this:
> ```sh
> cargo run --release -- -l 21 -u 24
> ```
