For convenience, we also provide rust bindings to the ICICLE library for the following primitives:

- MSM
- NTT
    - Forward NTT
    - Inverse NTT
- ECNTT
    - Forward ECNTT
    - Inverse NTT
- Scalar Vector Multiplication
- Point Vector Multiplication

A custom [build script][B_SCRIPT] is used to compile and link the ICICLE library. The environnement variable `ARCH_TYPE` is used to determine which GPU type the library should be compiled for and it defaults to `native` when it is not set allowing the compiler to detect the installed GPU type.

> NOTE: A GPU must be detectable and therefore installed if the `ARCH_TYPE` is not set.

Once you have your parameters set, run:

```sh
cargo build --release
```

You'll find a release ready library at `target/release/libicicle_utils.rlib`.

To benchmark and test the functionality available in RUST, run:

```
cargo bench
cargo test -- --test-threads=1
```

The flag `--test-threads=1` is needed because currently some tests might interfere with one another inside the GPU.