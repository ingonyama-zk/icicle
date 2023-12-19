### Rust Bindings

`icicle-core` defines all interfaces, macros and common methods.

`icicle-cuda-runtime` defines `DeviceContext` which can be used to manage a specific GPU as well as wrapping common CUDA methods.

`icicle-curves` implements all interfaces and macros from `icicle-core` for each curve. For example `icicle-bn254` implements curve `bn254`. Each curve has its own [build script](./icicle-curves/icicle-bn254/build.rs) which will build the CUDA libraries for that curve as part of the rust-toolchain build.

## Building a curve and running tests

Enter a curve implementation.

```
cd icicle-curves/icicle-bn254
```

To build 

```sh
cargo build --release
```

The build may take a while because we are also building the CUDA libraries for the selected curve.

To run benchmarks

```
cargo bench
```

To run test

```sh
cargo test -- --test-threads=1
```

The flag `--test-threads=1` is needed because currently some tests might interfere with one another inside the GPU.


### Example Usage

An example of using the Rust bindings library can be found in our [fast-danksharding implementation][FDI]

<!-- Begin Links -->
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
<!-- End Links -->
