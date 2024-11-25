# ICICLE example: Number Theoretic Transform (NTT) in Rust

## Key-Takeaway

`ICICLE` provides Rust bindings to CUDA-accelerated C++ implementation of [Number Theoretic Transform](https://github.com/ingonyama-zk/ingopedia/blob/master/src/fft.md).

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
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
# for METAL
./run.sh -d METAL -b /path/to/cuda/backend/install/dir
```
