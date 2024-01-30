# ICICLE example: Number Theoretic Transform (NTT) in Rust

## Key-Takeaway

`ICICLE` provides Rust bindings to CUDA-accelerated C++ implementation of [Number Theoretic Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md).

## Best Practices

In order to save time and setting up prerequisites manually, we recommend running this example in our [ZKContainer](../../ZKContainer.md).

## Usage

```rust
ntt::ntt(
  /* input slice */ scalars.as_slice(),
  /* NTT Direction */ ntt::NTTDir::kForward,
  /* NTT Configuration */ &cfg,
  /* output slice */ ntt_results.as_slice()
).unwrap();
```

In this example we use the `BN254` and `BLS12377` fields.

## What's in this example

1. Define the size of NTT.
2. Generate random inputs on-host
3. Set up the domain.
4. Configure NTT
5. Execute NTT on-device
6. Move the result on host
7. Compare results with arkworks

Running the example:

```sh
cargo run --release
```

You can add the `--feature profile` flag to measure times of both ICICLE and arkworks.

> [!NOTE]
> The default size is 2^20. You can change this by passing the `--size <size>` option. To change the size to 2^23, run the example like this:

```sh
cargo run --release -- -s 23
```

## Benchmarks

These benchmarks were run on a 16 core 24 thread i9-12900k CPU and an RTX 3090 Ti GPU

### Single BN254 NTT

| Library\Size | 2^19 | 2^20 | 2^21 | 2^22 | 2^23 |
|--------------|------|------|------|------|------|
| ICICLE | 1.263 ms | 2.986 ms | 4.651 ms | 9.308 ms | 18.618 ms |
| Arkworks | 138 ms | 290 ms | 611 ms | 1,295 ms | 2,715 ms |

### Single BLS12377 NTT

| Library\Size | 2^19 | 2^20 | 2^21 | 2^22 | 2^23 |
|--------------|------|------|------|------|------|
| ICICLE | 1.272 ms | 2.893 ms | 4.728 ms | 9.211 ms | 18.319 ms |
| Arkworks | 135 ms | 286 ms | 605 ms | 1,279 ms | 2,682 ms |
