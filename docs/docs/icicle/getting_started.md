
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

:::info
To specify the field, use the flag -DFIELD=field, where field can be one of the following: babybear, stark252, m31.

To specify a curve, use the flag -DCURVE=curve, where curve can be one of the following: bn254, bls12_377, bls12_381, bw6_761, grumpkin.
:::

:::tip
If you have access to cuda backend repo, it can be built along ICICLE frontend by adding the following to the cmake command
- `-DCUDA_BACKEND=local` # if you have it locally
- `-DCUDA_BACKEND=<commit|branch>` # to pull CUDA backend, given you have access
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

#### Features

By default, all [features](./libraries.md#supported-curves-and-operations) are enabled.

:::note
Installed backends may implement and register all APIs, therefore by default we include them in the frontend part too.
:::

To disable features, add the following to the cmake command.
- ntt: `-DNTT=OFF`
- msm: `-DMSM=OFF`
- g2 msm: `-DG2=OFF`
- ecntt: `-DECNTT=OFF`
- extension field: `-DEXT_FIELD=OFF`

### Rust: Build, Test, and Install

To build and test ICICLE in Rust, follow these steps:

1. **Navigate to the Rust bindings directory:**
```bash
cd wrappers/rust # or go to a specific field/curve 'cd wrappers/rust/icicle-fields/icicle-babybear'
```

2. **Build the Rust project:**
```bash
cargo build --release
```
By default, all [features](./libraries.md#supported-curves-and-operations) are enabled.
In rust we have the following features:
- `g2` for G2 MSM
- `ec_ntt` for ECNTT

They can be disabled by
```bash
cargo build --release --no-default-features # disable all features
cargo build --release --no-default-features --features "ec_ntt" # disable all except ec_ntt
cargo build --release --no-default-features --features "g2" # disable all except g2
```

:::tip
If you have access to cuda backend repo, it can be built along ICICLE frontend by using the following cargo features:
- `cuda_backend` : if the cuda backend resides in `icicle/backend/cuda`
- `pull_cuda_backend` : to pull main branch and build it
:::


3. **Run tests:**
```bash
cargo test # optional:  --no-default-features
```
:::note
Most tests assume a CUDA backend is installed and fail otherwise.
:::

4. **Install the library:**

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
# add other ICICLE crates here if need additional fields/curves
```

Can specify `branch = <branch-name>` or `tag = <tag-name>` or `rev = <commit-id>`.

To disable features:
```bash
icicle-bls12-377 = { path = "git = "https://github.com/ingonyama-zk/icicle.git", default-features = false, features = ["g2"] }
# or for g2 only
icicle-bls12-377 = { path = "git = "https://github.com/ingonyama-zk/icicle.git", default-features = false}
```


As explained above, the libs will be built and installed to `target/<buildmode>/deps/icicle` so you can easily link to them. Alternatively you can set `ICICLE_INSTALL_DIR` env variable for a custom install directory.
:::note
Make sure to install icicle libs when installing a library/application that depends on icicle.
:::

### Go: Build, Test, and Install (TODO)

## Install cuda backend

[Install CUDA Backend (and License)](./install_cuda_backend.md#installation)