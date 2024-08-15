
# Getting Started with ICICLE V3

This guide will help you get started with building, testing, and installing ICICLE, whether you're using C++, Rust, or Go. It also covers installation of the CUDA backend and important build options.

## Building and Testing ICICLE frontend

### C++: Build, Test, and Install (Frontend)

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
	To specify the field, use the flag -DFIELD=field, where field can be one of the following: babybear, stark252, m31.
	To specify a curve, use the flag -DCURVE=curve, where curve can be one of the following: bn254, bls12_377, bls12_381, bw6_761, grumpkin.
:::

3. **Build the project:**
   ```bash
   cmake --build build -j
   ```
   This is building the [libicicle_device](./libraries.md#icicle-device) and the [libicicle_field_babybear](./libraries.md#icicle-core) frontend lib that correspond to the field or curve.

4. **Link:**
   Link you application (or library) to ICICLE:
   ```cmake
   target_link_libraries(yourApp PRIVATE icicle_field_babybear icicle_device)
   ```


5. **Installation (optional):**
   To install the libs, specify the install prefix in the [cmake command](./getting_started.md#build-commands)
   `-DCMAKE_INSTALL_PREFIX=/install/dir/`. Default install path is `/usr/local` if not specified.
   Then after building, use cmake to install the libraries:
   ```
   cmake -S icicle_v3 -B build -DCMAKE_INSTALL_PREFIX=/path/to/install/dir/ -DCMAKE_BUILD_TYPE=Release -DFIELD=babybear
   cmake --build build -j # build ICICLE
   cmake --install build # install icicle in /path/to/install/dir/
   ```

6. **Run tests (optional):**
   Add `-DBUILD_TESTS=ON` to the [cmake command](./getting_started.md#build-commands) and build.
   Execute all tests
   ```bash
   cmake -S icicle_v3 -B build --DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DFIELD=babybear
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
:::note
Some tests assume a cuda backend exists and will fail otherwise.
:::
:::note
Each C++ test-suite is an executable that links to gtest.
:::

#### Build Flags

You can customize your ICICLE build with the following flags:

- `-DCPU_BACKEND=ON/OFF`: Enable or disable built it CPU backend. Default=ON.
- `-DBUILD_TESTS=ON/OFF`: Enable or disable tests. Default=OFF.
- `-DCMAKE_INSTALL_PREFIX=/install/dir`: Specify install dir. Default=/usr/local.

### Rust: Build, Test, and Install

To build and test ICICLE in Rust, follow these steps:

1. **Navigate to the Rust bindings directory:**
   ```bash
   cd wrappers/rust # or go to a specific field/curve 'cd wrappers/rust/icicle-fields/icicle-babybear'
   ```

2. **Build the Rust project:**
   ```bash
   TODO wht about features? Now it doesn't make sense to disable so??
   cargo build --release
   ```

3. **Run tests:**
   ```bash
   cargo test
   ```

4. **Install the library:**: The libraries are installed to the `target/<buildmode>/deps/icicle` dir by default. For custom install dir. define the env variable `export ICICLE_INSTALL_DIR=/path/to/install/dir` before building or via cargo:
   ```bash
   TODO support cargo install
   cargo install --path /path/to/install/dir
   ```

#### Use as cargo dependency

In cargo.toml, specify the ICICLE libs to use:

```bash
TODO fix paths

[dependencies]
icicle-runtime = { path = "git = "https://github.com/ingonyama-zk/icicle.git"" }
icicle-core = { path = "git = "https://github.com/ingonyama-zk/icicle.git"" }
icicle-bls12-377 = { path = "git = "https://github.com/ingonyama-zk/icicle.git" }
```

The libs will be built and installed to `target/<buildmode>/deps/icicle` so you can easily link to them. Alternatively you can set `ICICLE_INSTALL_DIR` env variable to have it installed elsewhere.
:::note
Make sure to have the icicle libs available when deploying an application that depends on icicle shared libs.
:::

### Go: Build, Test, and Install
TODO

## Install cuda backend

[Install CUDA Backend (and License)](./install_cuda_backend.md#installation)