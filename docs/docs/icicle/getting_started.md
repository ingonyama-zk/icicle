
# Getting Started with ICICLE V3

Welcome to the ICICLE V3 documentation! This guide will help you get started with building, testing, and installing ICICLE, whether you're using C++, Rust, or Go. It also covers installation of the CUDA backend and important build options.

## Building and Testing ICICLE

### C++: Build, Test, and Install (Frontend and Backend)

ICICLE can be built and tested in C++ using CMake. The build process is straightforward, but there are several flags you can use to customize the build for your needs.

#### Build Commands

1. **Clone the ICICLE repository:**
   ```bash
   git clone https://github.com/ingonyama-zk/icicle.git
   cd icicle
   ```

2. **Configure the build:**
   ```bash
   (TODO replace icicle_v3 with icicle)
   mkdir -p build && rm -rf build/*
   cmake -S icicle_v3 -B build -DCMAKE_BUILD_TYPE=Release -DFIELD=babybear
   ```

:::note
	To specify the field, use the flag -DFIELD=field, where field can be one of the following: babybear, stark252.
	To specify a curve, use the flag -DCURVE=curve, where curve can be one of the following: bn254, bls12_377, bls12_381, bw6_761, grumpkin.
:::

3. **Build the project:**
   ```bash
   cmake --build build -j
   ```
   This is building the [libicicle_device](./libraries.md#icicle-device) and the [libicicle_field_babybear](./libraries.md#icicle-core) frontend lib that correspond to the field or curve.

4. **Link:**
   ```cmake
   target_link_libraries(yourApp PRIVATE icicle_field_babybear icicle_device)
   ```

#### optional Commands

5. **Installation**
   To install the libs, specify the install prefix in the [cmake command](./getting_started.md#build-commands)
   `-DCMAKE_INSTALL_PREFIX=/install/dir/`. Default install path is `/usr/local`.
   Then after building, use cmake to install `cmake --install build`

6. **Run tests:**
   Add `-DCMAKE_BUILD_TESTS=ON` to the [cmake command](./getting_started.md#build-commands) and build.
   Execute all tests
   ```bash
   cd build/tests
   ctest
   ```
   or choose the test-suite and possible test
   ```bash
   ./build/tests/test_field_api # or another test suite
   # can specify test using regex
   ./build/tests/test_field_api --gtest_filter="*ntt*"
   ```
:::note
Some yests assume a cuda backend exists and may fail if they don't.
:::

#### Build Flags

You can customize your ICICLE build with the following flags:

- `-DCPU_BACKEND=ON/OFF`: Enable or disable built it CPU backend. Default=ON.
- `-DBUILD_TESTS=ON/OFF`: Enable or disable tests. Default=OFF.
- `-DCMAKE_INSTALL_PREFIX=/install/dir`: Specify install dir. Default=/usr/local.


### Rust: Build, Test, and Install

#### Rust

To build and test ICICLE in Rust, follow these steps:

1. **Navigate to the Rust bindings directory:**
   ```bash
   cd wrappers/rust
   ```

2. **Build the Rust project:**
   ```bash
   cargo build --release
   ```

3. **Run tests:**
   ```bash
   cargo test
   ```

4. **Install the library:**
   ```bash
   cargo install --path .
   ```

#### Rust Cargo Build and Install

To build ICICLE as dep using Cargo:

```bash
[dependencies]
TODO fix path
icicle-runtime = { path = "crate_path" }
icicle-core = { path = "crate_path" }
icicle-bn254 = { path = "crate_path" }
```

The libs will be built and installed to `target/release/deps/icicle` so you can easily link to them. Alternatively you can set `ICICLE_INSTALL_DIR` env variable to have it installed elsewhere.
:::note
Make sure to have the icicle libs avaialble when deploying an application that depends on icicle shared libs.
:::



## Installation

[Install CUDA Backend (and License)](./install_cuda_backend.md#installation)

### Install ICICLE Frontend

ICICLE frontend installation can vary based on your needs. You can install the frontend libraries using standard package managers for your language (e.g., `cargo` for Rust, `go get` for Go), or you can manually build and install from source as outlined above.



#### Go Get Install

TODO