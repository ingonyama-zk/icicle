# Rust bindings

Rust bindings allow you to use ICICLE as a rust library.

`icicle-core` defines all interfaces, macros and common methods.

`icicle-cuda-runtime` defines DeviceContext which can be used to manage a specific GPU as well as wrapping common CUDA methods.

`icicle-curves` implements all interfaces and macros from icicle-core for each curve. For example icicle-bn254 implements curve bn254. Each curve has its own build script which will build the CUDA libraries for that curve as part of the rust-toolchain build.

## Using ICICLE Rust bindings in your project

Simply add the following to your `Cargo.toml`.

```
# GPU Icicle integration
icicle-cuda-runtime = { git = "https://github.com/ingonyama-zk/icicle.git" }
icicle-core = { git = "https://github.com/ingonyama-zk/icicle.git" }
icicle-bn254 = { git = "https://github.com/ingonyama-zk/icicle.git" }
```

`icicle-bn254` being the curve you wish to use and `icicle-core` and `icicle-cuda-runtime` contain ICICLE utilities and CUDA wrappers.

If you wish to point to a specific ICICLE branch add `branch = "<name_of_branch>"` or `tag = "<name_of_tag>"` to the ICICLE dependency. For a specific commit add `rev = "<commit_id>"`.

When you build your project ICICLE will be built as part of the build command.

# How do the rust bindings work?

The rust bindings are just rust wrappers for ICICLE Core static libraries which can be compiled. We integrate the compilation of the static libraries into rusts toolchain to make usage seamless and easy. This is achieved by [extending rusts build command](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-curves/icicle-bn254/build.rs).

```rust
use cmake::Config;
use std::env::var;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let out_dir = Config::new("../../../../icicle")
                .define("BUILD_TESTS", "OFF") //TODO: feature
                .define("CURVE", "bn254")
                .define("CMAKE_BUILD_TYPE", "Release")
                .build_target("icicle")
                .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
```
