
# Install and use ICICLE

## Overview

This page describes the content of a release and how to install and use it.
Icicle binaries are released for multiple Linux distributions, including Ubuntu 20.04, Ubuntu 22.04, and CentOS 7.

:::note
Future releases will also include MacOS and other systems.
:::

## Content of a Release

Each Icicle release includes a tar file, named `icicle30-<distribution>.tar.gz`, where `icicle30` stands for version 3.0. This tar contains icicle-frontend build artifacts  and headers for a specific distribution. The tar file includes the following structure:

- **`./icicle/include/`**: This directory contains all the necessary header files for using the Icicle library from C++.
- **`./icicle/lib/`**:
  - **Icicle Libraries**: All the core Icicle libraries are located in this directory. Applications linking to Icicle will use these libraries.
  - **Backends**: The `./icicle/lib/backend/` directory houses backend libraries, including the CUDA backend (not included in this tar).

- **CUDA backend** comes as separate tar `icicle30-<distribution>-cuda122.tar.gz`
  - per distribution, for icicle-frontend V3.0 and CUDA 12.2.

## installing and using icicle

1. **Extract the Tar Files**:
   - Download (TODO link to latest release) the appropriate tar files for your distribution (Ubuntu 20.04, Ubuntu 22.04, or CentOS 7).
   - Extract it to your desired location:
     ```bash
     # install the frontend part (Can skip for Rust)
     tar -xzvf icicle30-<distribution>.tar.gz -C /opt/ # or other non-default install directory
     # install CUDA backend (Required for all programming-languages that want to use CUDA backend)
     tar -xzvf icicle30-<distribution>-cuda122.tar.gz -C /opt/ # or other non-default install directory
     ```

    - Note that you may install to any directory and you need to make sure it can be found by the linker at runtime.
    - Default location is `/opt`

:::tip
You can install anywhere and use a link so that it can be easily found as if in the default directory.
:::

2. **Linking Your Application**:

  Apps need to link to the ICICLE device library and in addition link to each field or curve libraries. The backend libraries are dynamically loaded at runtime, so not linking to them.

  **C++**
   - When compiling your C++ application, link against the Icicle libraries found in `/opt/icicle/lib` or other location:
     ```bash
     g++ -o myapp myapp.cpp -I/opt/icicle/include -L/opt/icicle/lib -licicle_device -licicle_field_bn254 -licicle_curve_bn254 -Wl,-rpath,/opt/icicle/lib/
     ```

    - Or via cmake
    ```bash
    # Include directories
    include_directories(/path/to/install/dir/icicle/include)
    # Library directories
    link_directories(/path/to/install/dir/icicle/lib/)
    # Add the executable
    add_executable(example example.cpp)
    # Link the libraries
    target_link_libraries(example icicle_device icicle_field_bn254 icicle_curve_bn254)
    # Set the RPATH so linker finds icicle libs at runtime
    set_target_properties(example PROPERTIES
        BUILD_RPATH /path/to/install/dir/icicle/lib/
        INSTALL_RPATH /path/to/install/dir/icicle/lib/)
    ```

    :::tip
    If you face linkage issues, try `ldd myapp` to see the runtime deps. If ICICLE libs are not found, you need to add the install directory to the search path of the linker. In a development env you can do that using the env variable `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/icicle/lib` or similar (for non linux). For deployment, make sure it can be found and avoid `LD_LIBRARY_PATH`.

    Alternatively you can embed the search path on the app as an `rpath` by adding `-Wl,-rpath,/path/to/icicle/lib/`. Now the linker will search there too.
    :::

  **Rust**
     - When building the icicle crates, icicle frontend libs are built from source, in addition to the rust bindings. They are installed to `target/<buildtype>/deps/icile` and cargo will link correctly. Note that you still need to install CUDA backend if you have a CUDA GPU.
     - Simply use `cargo build` or `cargo run` and it should link to icicle libs.      

  **Go** - TODO

:::warning when deploying an application (either C++, Rust or Go), you must make sure to either deploy the icicle libs (in Rust it's in `target/<buildtype>/deps/icile` or the preinstalled ones) along the application binaries (as tar, docker image, package manager installer or else) or make sure to install icicle (and the backend) on the target machine. Otherwise the target machine will have linkage issues.
:::

## Backend Loading

The Icicle library dynamically loads backend libraries at runtime. By default, it searches for backends in the following order:

1. **Environment Variable**: If the `ICICLE_BACKEND_INSTALL_DIR` environment variable is defined, Icicle will prioritize this location.
2. **Default Directory**: If the environment variable is not set, Icicle will search in the default directory `/opt/icicle/lib/backend`.

:::warning
Make sure to load a backend that is compatible to the frontend version. CUDA backend libs are forward compatible with newer frontends (e.g. CUDA-backend-3.0 works with ICICLE-3.2). The opposite is not guaranteed.
:::

If you install in a custom dir, make sure to set `ICICLE_BACKEND_INSTALL_DIR`:
```bash
ICICLE_BACKEND_INSTALL_DIR=path/to/icicle/lib/backend/ myapp # for an executable maypp
ICICLE_BACKEND_INSTALL_DIR=path/to/icicle/lib/backend/ cargo run # when using cargo
```

Then to load backend from ICICLE_BACKEND_INSTALL_DIR or `/opt/icicle/lib/backend` in your application:

**C++**
```cpp
extern "C" eIcicleError icicle_load_backend_from_env_or_default();
```
**Rust**
```rust
pub fn load_backend_from_env_or_default() -> Result<(), eIcicleError>;
```
**Go**
```go
TODO
```

### Custom Backend Loading

If you need to load a backend from a custom location at any point during runtime, you can call the following function:

**C++**
```cpp
extern "C" eIcicleError icicle_load_backend(const char* path, bool is_recursive);
```
- **`path`**: The directory where the backend libraries are located.
- **`is_recursive`**: If `true`, the function will search for backend libraries recursively within the specified path.

**Rust**
```rust
  pub fn load_backend(path: &str) -> Result<(), eIcicleError>; // OR
  pub fn load_backend_non_recursive(path: &str) -> Result<(), eIcicleError>;
```
- **`path`**: The directory where the backend libraries are located.

**Go**
```go
TODO
```

:::note
When loading from the backends dir, you may see the following:
```
[INFO] Attempting to load: some/path/icicle/lib/backend/bls12_381/cuda/libicicle_cuda_curve_bls12_381.so
[INFO] Failed to load some/path/icicle/lib/backend/bls12_381/cuda/libicicle_cuda_curve_bls12_381.so: libicicle_curve_bls12_381.so: cannot open shared object file: No such file or directory
```

In this case the cuda backend for bls12_381 curve failed to load since it cannot find the corresponding frontend. This should not happen if the bls12_381 frontend is linked to the application.
Also note that if the frontend libs are installed and found by the linker, they will be loaded as well. This is not a problem except for loading unused libs to the process. Can avoid it by specifying a more specific path but make sure to load all required libs.
:::
