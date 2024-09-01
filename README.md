# ICICLE

<div align="center">ICICLE is a high-performance cryptographic acceleration library designed to optimize cryptographic computations across various hardware platforms, including CPUs, GPUs, and other accelerators.</div>

<p align="center">
  <img alt="ICICLE" width="300" height="300" src="https://user-images.githubusercontent.com/2446179/223707486-ed8eb5ab-0616-4601-8557-12050df8ccf7.png"/>
</p>
<p align="center">
  <a href="https://discord.gg/EVVXTdt6DF">
    <img src="https://img.shields.io/discord/1063033227788423299?logo=discord" alt="Chat with us on Discord">
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=Ingo_zk">
    <img src="https://img.shields.io/twitter/follow/Ingo_zk?style=social&logo=twitter" alt="Follow us on Twitter">
  </a>
  <a href="https://github.com/ingonyama-zk/icicle/releases">
    <img src="https://img.shields.io/github/v/release/ingonyama-zk/icicle" alt="GitHub Release">
  </a>
</p>


## Background

Zero Knowledge Proofs (ZKPs) are considered one of the greatest achievements of modern cryptography. Accordingly, ZKPs are expected to disrupt a number of industries and will usher in an era of trustless and privacy preserving services and infrastructure.

We believe that ICICLE will be a cornerstone in the acceleration of ZKPs:

- Versatility: ICICLE supports multiple hardware platforms, making it adaptable to various computational environments.
- Efficiency: ICICLE is designed to leverage the parallel nature of ZK computations, whether on GPUs, CPUs, or other accelerators.
- Scalability: ICICLE provides an easy-to-use and scalable solution for developers, allowing them to optimize cryptographic operations with minimal effort.

## Getting Started

ICICLE is a versatile cryptographic acceleration library with support for multiple platforms. This guide will help you get started with ICICLE in C++, Rust, and Go.

> [!NOTE]
> Developers: We highly recommend reading our [documentation](https://dev.ingonyama.com/) for a comprehensive understanding of ICICLE’s capabilities.

> [!TIP]
> Try out ICICLE by running some [examples](https://github.com/ingonyama-zk/icicle/tree/yshekel/V3/examples) available in C++, Rust, and Go bindings. Check out our install-and-use examples in [C++](https://github.com/ingonyama-zk/icicle/tree/yshekel/V3/examples/c%2B%2B/install-and-use-icicle), [Rust](https://github.com/ingonyama-zk/icicle/tree/yshekel/V3/examples/rust/install-and-use-icicle) and [Go](TODO)

### Prerequisites

- Any Compatible Hardware: ICICLE supports various hardware, including CPUs, Nvidia GPUs, and other accelerators.
- [CMake]((https://cmake.org/files/)), Version 3.18 or above. Latest version recommended. Required only if building from source.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) Required only if using NVIDIA GPUs (version 12.0 or newer).

> [!NOTE]
> For older GPUs that only support CUDA 11, ICICLE may still function, but official support is for CUDA 12 or newer.

### Accessing Hardware

If you don't have access to an Nvidia GPU we have some options for you.

Checkout [Google Colab](https://colab.google/). Google Colab offers a free [T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) instance and ICICLE can be used with it, reference this guide for setting up your [Google Colab workplace][GOOGLE-COLAB-ICICLE].

If you require more compute and have an interesting research project, we have [bounty and grant programs][GRANT_PROGRAM].

## Building ICICLE from source

ICICLE provides build systems for C++, Rust, and Go. Each build system incorporates the core ICICLE library, which contains the essential cryptographic primitives.

Refer to [Getting started page](https://dev.ingonyama.com/icicle/getting_started) and [Build From Source](https://dev.ingonyama.com/icicle/build_from_source) for full details about building and using ICICLE.

> [!WARNING]
> Ensure ICICLE libraries are installed correctly when building or installing a library/application that depends on ICICLE so that they can be located at runtime.

### Rust

In cargo.toml, specify the ICICLE libs to use:

```bash
[dependencies]
icicle-runtime = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
icicle-core = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
icicle-babybear = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
# add other ICICLE crates here if need additional fields/curves
```

You can specify `branch=branch-name`, `tag=tag-name`, or `rev=commit-id`.

Build the Rust project:

```bash
cargo build --release
```

### Go

TODO

### C++

ICICLE can be built and tested in C++ using CMake. The build process is straightforward, but there are several flags you can use to customize the build for your needs.

**Clone the ICICLE repository:**

```bash
git clone https://github.com/ingonyama-zk/icicle.git
cd icicle
```

**Configure the build:**

```bash
mkdir -p build && rm -rf build/*
cmake -S icicle -B build -DFIELD=babybear
```

> [!NOTE]
> To specify the field, use the flag -DFIELD=field, where field can be one of the following: babybear, stark252, m31.
> To specify a curve, use the flag -DCURVE=curve, where curve can be one of the following: bn254, bls12_377, bls12_381, bw6_761, grumpkin.

**Build the project:**

```bash
cmake --build build -j # -j is for multi-core compilation
```

**Link you application (or library) to ICICLE:**

```cmake
target_link_libraries(yourApp PRIVATE icicle_field_babybear icicle_device)
```

**Install (optional):**

To install the libs, specify the install prefix `-DCMAKE_INSTALL_PREFIX=/install/dir/`. Then after building, use cmake to install the libraries:
```
cmake -S icicle -B build -DFIELD=babybear -DCMAKE_INSTALL_PREFIX=/path/to/install/dir/
cmake --build build -j # build
cmake --install build # install icicle to /path/to/install/dir/
```

**Run tests (optional):**

Add `-DBUILD_TESTS=ON` to the cmake command, build and execute tests:
```bash
cmake -S icicle -B build -DFIELD=babybear -DBUILD_TESTS=ON
cmake --build build -j
cd build/tests
ctest
```
or choose the test-suite
```bash
./build/tests/test_field_api # or another test suite
# can specify tests using regex. For example for tests with ntt in the name:
./build/tests/test_field_api --gtest_filter="*ntt*"
```

>  [!NOTE]
> Most tests assume a CUDA backend exists and will fail otherwise if a CUDA device is not found.

**Build Flags:**

You can customize your ICICLE build with the following flags:

- `-DCPU_BACKEND=ON/OFF`: Enable or disable built-in CPU backend. `default=ON`.
- `-DCMAKE_INSTALL_PREFIX=/install/dir`: Specify install directory. `default=/usr/local`.
- `-DBUILD_TESTS=ON/OFF`: Enable or disable tests. `default=OFF`.
- `-DBUILD_BENCHMARKS=ON/OFF`: Enable or disable benchmarks. `default=OFF`.

## Install cuda backend

To install the CUDA backend

1. [Download the release binaries](https://github.com/ingonyama-zk/icicle/releases/).
2. Install it, by extracting the binaries to `/opt/` or any other custom install path.
3. In your application, load the cuda backend and select a CUDA device.
4. All subsequent API will now use the selected device.

C++:

```cpp
#include "icicle/runtime.h"

// Load the installed backend
eIcicleError result = icicle_load_backend_from_env_or_default();
// or load it programmatically
eIcicleError result = icicle_load_backend("/path/to/backend/installdir", true);

// Select CUDA device
icicle::Device device = {"CUDA", 0 /*gpu-id*/};
eIcicleError result = icicle_set_device(device);

// Any call will now execute on GPU-0
```

Rust:

```rust
use icicle_runtime::{runtime, Device};

runtime::load_backend_from_env_or_default().unwrap();
// or load programmatically
runtime::load_backend("/path/to/backend/installdir").unwrap();
// Select CUDA device
let device = Device::new("CUDA", 1 /*gpu-id*/);
icicle_runtime::set_device(&device).unwrap();

// Any call will now execute on GPU-1
```

Go:

TODO


Full details can be found in our [docs](https://dev.ingonyama.com/icicle/https://dev.ingonyama.com/icicle/getting_started)

## Contributions

Join our [Discord Server][DISCORD] and find us on the icicle channel. We will be happy to work together to support your use case and talk features, bugs and design.

### Development Contributions

If you are changing code, please make sure to change your [git hooks path][HOOKS_DOCS] to the repo's [hooks directory][HOOKS_PATH] by running the following command:

```sh
git config core.hooksPath ./scripts/hooks
```

In case `clang-format` is missing on your system, you can install it  using the following command:

```sh
sudo apt install clang-format
```

You will also need to install [codespell](https://github.com/codespell-project/codespell?tab=readme-ov-file#installation) to check for typos.

This will ensure our custom hooks are run and will make it easier to follow our coding guidelines.

### Hall of Fame

- [Robik](https://github.com/robik75), for his ongoing support and mentorship
- [liuxiao](https://github.com/liuxiaobleach), for being a top notch bug smasher
- [gkigiermo](https://github.com/gkigiermo), for making it intuitive to use ICICLE in Google Colab
- [nonam3e](https://github.com/nonam3e), for adding Grumpkin curve support into ICICLE
- [alxiong](https://github.com/alxiong), for adding warmup for CudaStream
- [cyl19970726](https://github.com/cyl19970726), for updating go install source in Dockerfile
- [PatStiles](https://github.com/PatStiles), for adding Stark252 field

## Help & Support

For help and support talk to our devs in our discord channel ["ICICLE"](https://discord.gg/EVVXTdt6DF) or contact us at support@ingonyama.com.

## License

ICICLE frontend is distributed under the terms of the MIT License.

> [!NOTE]
> ICICLE backends, excluding the CPU backend, are distributed under a special license and are not covered by the MIT license.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[BLS12-381]: ./icicle/curves/
[BLS12-377]: ./icicle/curves/
[BN254]: ./icicle/curves/
[BW6-671]: ./icicle/curves/
[Grumpkin]: ./icicle/curves/
[babybear]: ./icicle/fields/
[stark252]: ./icicle/fields/
[m31]: ./icicle/fields/
[LMIT]: ./LICENSE
[DISCORD]: https://discord.gg/Y4SkbDf2Ff
[googletest]: https://github.com/google/googletest/
[HOOKS_DOCS]: https://git-scm.com/docs/githooks
[HOOKS_PATH]: ./scripts/hooks/
[GOOGLE-COLAB-ICICLE]: https://dev.ingonyama.com/icicle/colab-instructions
[GRANT_PROGRAM]: https://medium.com/@ingonyama/icicle-for-researchers-grants-challenges-9be1f040998e
[ICICLE-CORE]: ./icicle/
[ICICLE-RUST]: ./wrappers/rust/
[ICICLE-GO]: ./wrappers/golang/
[ICICLE-CORE-README]: ./icicle/README.md
[ICICLE-RUST-README]: ./wrappers/rust/README.md
[ICICLE-GO-README]: ./wrappers/golang/README.md
[documentation]: https://dev.ingonyama.com/icicle/overview
[examples]: ./examples/

<!-- End Links -->
