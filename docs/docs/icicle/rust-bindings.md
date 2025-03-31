# Rust bindings

Rust bindings allow you to use ICICLE as a rust library.

`icicle-core` defines all interfaces, macros and common methods.

`icicle-runtime` contains runtime APIs for memory management, stream management and more.

`icicle-curves` / `icicle-fields` implement all interfaces and macros from icicle-core for each curve. For example icicle-bn254 implements curve bn254. Each curve has its own build script which will build the ICICLE libraries for that curve as part of the rust-toolchain build.

## Documentation

- [Core Traits and Types](rust-bindings/core.md) - Documentation for core traits and types used throughout the bindings
- [Field Operations](rust-bindings/vec-ops.md) - Documentation for field arithmetic operations
- [Curve Operations](rust-bindings/msm.md) - Documentation for curve operations
- [Hash Functions](rust-bindings/hash.md) - Documentation for hash functions
- [Merkle Trees](rust-bindings/merkle.md) - Documentation for Merkle tree operations
- [NTT Operations](rust-bindings/ntt.md) - Documentation for Number Theoretic Transform operations
- [ECNTT Operations](rust-bindings/ecntt.md) - Documentation for Elliptic Curve Number Theoretic Transform operations
- [Polynomial Operations](rust-bindings/polynomials.md) - Documentation for polynomial operations
- [Program Operations](rust-bindings/program.md) - Documentation for program operations
- [Sumcheck Protocol](rust-bindings/sumcheck.md) - Documentation for the sumcheck protocol
- [Multi-GPU Support](rust-bindings/multi-gpu.md) - Documentation for multi-GPU operations

## Using ICICLE Rust bindings in your project

Simply add the following to your `Cargo.toml`.

```toml
# GPU Icicle integration
icicle-runtime = { git = "https://github.com/ingonyama-zk/icicle.git" }
icicle-core = { git = "https://github.com/ingonyama-zk/icicle.git" }
icicle-bn254 = { git = "https://github.com/ingonyama-zk/icicle.git" }
```

`icicle-bn254` being the curve you wish to use and `icicle-core` and `icicle-runtime` contain ICICLE utilities and CUDA wrappers.

If you wish to point to a specific ICICLE branch add `branch = "<name_of_branch>"` or `tag = "<name_of_tag>"` to the ICICLE dependency. For a specific commit add `rev = "<commit_id>"`.

When you build your project ICICLE will be built as part of the build command.

## How do the rust bindings work?

The rust bindings are rust crates that wrap the ICICLE Core libraries (C++). Each crate can wrap one or more ICICLE core libraries. They are built too when building the crate.

:::note
Since ICICLE V3, core libraries are shared-libraries. This means that they must be installed in a directory that can be found by the linker. In addition, installing an application that depends on ICICLE must make sure to install ICICLE or have it installed on the target machine.
:::
