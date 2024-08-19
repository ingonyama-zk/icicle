
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
   mkdir -p build && rm -rf build/*
   cmake -S icicle -B build -DFIELD=babybear
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
   `-DCMAKE_INSTALL_PREFIX=/install/dir/`. Default install path on linux is `/usr/local` if not specified. For other systems it may differ. The cmake command will print it to the log
   ```
   -- CMAKE_INSTALL_PREFIX=/install/dir/for/cmake/install
   ```
   Then after building, use cmake to install the libraries:
   ```
   cmake -S icicle -B build -DFIELD=babybear -DCMAKE_INSTALL_PREFIX=/path/to/install/dir/
   cmake --build build -j # build
   cmake --install build # install icicle to /path/to/install/dir/
   ```

6. **Run tests (optional):**
   Add `-DBUILD_TESTS=ON` to the [cmake command](./getting_started.md#build-commands) and build.
   Execute all tests
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
:::note
Most tests assume a cuda backend exists and will fail otherwise if cannot find a CUDA device.
:::

#### Build Flags

You can customize your ICICLE build with the following flags:

- `-DCPU_BACKEND=ON/OFF`: Enable or disable built-in CPU backend. `default=ON`.
- `-DCMAKE_INSTALL_PREFIX=/install/dir`: Specify install directory. `default=/usr/local`.
- `-DBUILD_TESTS=ON/OFF`: Enable or disable tests. `default=OFF`.
- `-DBUILD_BENCHMARKS=ON/OFF`: Enable or disable benchmarks. `default=OFF`.

### Rust: Build, Test, and Install

To build and test ICICLE in Rust, follow these steps:

1. **Navigate to the Rust bindings directory:**
   ```bash
   cd wrappers/rust # or go to a specific field/curve 'cd wrappers/rust/icicle-fields/icicle-babybear'
   ```

2. **Build the Rust project:**
TODO what about features? Now it doesn't make sense to disable features.
   ```bash   
   cargo build --release
   ```

4. **Run tests:**
   ```bash
   cargo test
   ```
:::note
Most tests assume a CUDA backend is installed and fail otherwise.
:::

5. **Install the library:**

By default, the libraries are installed to the `target/<buildmode>/deps/icicle` dir. For custom install dir. define the env variable:
```bash
export ICICLE_INSTALL_DIR=/path/to/install/dir
```

(TODO: cargo install ?)

#### Use as cargo dependency

In cargo.toml, specify the ICICLE libs to use:

```bash
[dependencies]
icicle-runtime = { path = "git = "https://github.com/ingonyama-zk/icicle.git"" }
icicle-core = { path = "git = "https://github.com/ingonyama-zk/icicle.git"" }
icicle-bls12-377 = { path = "git = "https://github.com/ingonyama-zk/icicle.git" }
# add other ICICLE crates here if need aditional fields/curves
```

:::note
Can specify `branch = <branch-name>` or `tag = <tag-name>` or `rev = <commit-id>`.
:::

As explained above, the libs will be built and installed to `target/<buildmode>/deps/icicle` so you can easily link to them. Alternatively you can set `ICICLE_INSTALL_DIR` env variable for a custom install directory.
:::note
Make sure to isntall the icicle libs when installing a library/application that depends on icicle.
:::

### Go: Build, Test, and Install (TODO)

## Install cuda backend

[Install CUDA Backend (and License)](./install_cuda_backend.md#installation)