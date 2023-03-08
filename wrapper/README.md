# ICICLE

Rust bindings for ZK GPU primitives.

## Build and usage

> NOTE: NVCC and rust are prerequisites for building.

A custom [build script][B_SCRIPT] is used to compile and link the CUDA/cpp code. Make sure to change the `arch` flag depending on your GPU type or leave it as `native` for the compiler to detect the installed GPU.

> NOTE: In order to specify or define which curve to use, refer to [cuda/README.md][CUDA_USAGE]

Once you have your parameters set, run:

```sh
cargo build --release
```

You'll find a release ready library at `target/release/libicicle_utils.rlib`

### Example Usage

An example of using the Rust bindings library can be found in our [fast-danksharding implementation][FDI]

## API

- MSM
- NTT
    - Forward NTT
    - Inverse NTT
- ECNTT
    - Forward ECNTT
    - Inverse NTT
- Scalar Vector Multiplication
- Point Vector Multiplication

## Contributing

If you would like to contribute with code, check the [CONTRIBUTING.md][CONT] file for further info about the development environment.

## License

ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[B_SCRIPT]: ./ingo-cuda-utils/build.rs
[CUDA_USAGE]: ../cuda/README.md#usage
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
[CONT]: ./CONTRIBUTING.md
[LMIT]: ./LICENSE-MIT
<!-- End Links -->