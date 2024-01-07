# ICICLE example: MultiScalar Multiplication (MSM) in Rust

`ICICLE` provides Rust bindings to CUDA-accelerated C++ implementation of [Multi-Scalar Multiplication](https://github.com/ingonyama-zk/ingopedia/blob/master/src/msm.md).

## Best Practices

In order to save time and setting up prerequisites manually, we recommend running this example in our [ZKContainer™️](../../ZKContainer.md).

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
cargo run --release
```

You can add the `--feature arkworks,profile` flag to measure times of both ICICLE and arkworks.

> [!NOTE]
> The default sizes are 2^19 - 2^23. You can change this by passing the `--lower_bound_log_size <size> --upper_bound_log_size <size>` options. To change the size range to 2^21 - 2^24, run the example like this:
> ```sh
> cargo run --release -- -l 21 -u 24
> ```

## Benchmarks

These benchmarks were run on a 16 core 24 thread i9-12900k CPU and an RTX 3090 Ti GPU

### Single BN254 MSM
| Library\Size | 2^19 | 2^20 | 2^21 | 2^22 | 2^23 |
|--------------|------|------|------|------|------|
| ICICLE | 10 ms | 11 ms | 21 ms | 39 ms | 77 ms |
| Arkworks | 284 ms | 540 ms | 1,152 ms | 2,320 ms | 4,491 ms |

### Single BLS12377 MSM
| Library\Size | 2^19 | 2^20 | 2^21 | 2^22 | 2^23 |
|--------------|------|------|------|------|------|
| ICICLE | 9 ms | 14 ms | 25 ms | 48 ms | 93 ms |
| Arkworks | 490 ms | 918 ms | 1,861 ms | 3,624 ms | 7,191 ms |
