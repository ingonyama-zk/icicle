# Getting started Guide

## Overview

This guide will walk you through the entire process of building, testing, and installing ICICLE using your preferred programming language—C++, Rust, or Go. Whether you're deploying on a CPU or leveraging CUDA for accelerated performance, this guide provides comprehensive instructions to get you started. It also outlines the typical workflow for a user, including key installation steps:


1. **Install ICICLE or build it from source**: This is explained in this guide. For building from source, refer to the [Build from Source page](./build_from_source.md).
2. **Follow the [Programmer’s Guide](./programmers_guide/general.md)**: Learn how to use ICICLE APIs.  
3. **Start using ICICLE APIs on your CPU**: Your application will now use ICICLE on the CPU.
4. **Accelerate your application on a GPU**: [install the CUDA backend](./install_cuda_backend.md),  load it, and select it in your application ([C++](./programmers_guide/cpp.md#loading-a-backend),[Rust](./programmers_guide/rust.md#loading-a-backend), [Go](./programmers_guide/go.md#loading-a-backend)).
5. **Run on the GPU**: Once the GPU backend is selected, all subsequent API calls will execute on the GPU.
6. **Optimize for multi-GPU environments**: Refer to the [Multi-GPU](./multi-device.md) Guide to fully utilize your system’s capabilities.  
7. **Review memory management**: Revisit the [Memory Management section](./programmers_guide/general.md#device-abstraction) to allocate memory on the device efficiently and try to keep data on the GPU as long as possible.  
   

The rest of this page details the content of a release, how to install it, and how to use it. ICICLE binaries are released for multiple Linux distributions, including Ubuntu 20.04, Ubuntu 22.04, RHEL 8, and RHEL 9.

:::note
Future releases will also include support for macOS and other systems.
:::

## Content of a Release

Each ICICLE release includes a tar file named `icicle30-<distribution>.tar.gz`, where `icicle30` indicates version 3.0. This tar file contains ICICLE frontend build artifacts and headers for a specific distribution. The tar file structure includes:

- **`./icicle/include/`**: This directory contains all the necessary header files for using the ICICLE library from C++.
- **`./icicle/lib/`**:
  - **Icicle Libraries**: All the core ICICLE libraries are located in this directory. Applications linking to ICICLE will use these libraries.
  - **Backends**: The `./icicle/lib/backend/` directory houses backend libraries, including the CUDA backend (not included in this tar).

- **CUDA backend** comes as separate tar `icicle30-<distribution>-cuda122.tar.gz`
  - per distribution, for ICICLE-frontend v3.0 and CUDA 12.2.

## Installing and using ICICLE

- [Full C++ example](https://github.com/ingonyama-zk/icicle/tree/main/examples/c++/install-and-use-icicle)
- [Full Rust example](https://github.com/ingonyama-zk/icicle/tree/main/examples/rust/install-and-use-icicle)
- [Full Go example](https://github.com/ingonyama-zk/icicle/tree/main/examples/golang/install-and-use-icicle)

1. **Extract and install the Tar Files**:
   - [Download](https://github.com/ingonyama-zk/icicle/releases) the appropriate tar files for your distribution (Ubuntu 20.04, Ubuntu 22.04, or UBI 8,9 for RHEL compatible binaries).
   - **Frontend libs and headers** should be installed in default search paths (such as `/usr/lib` and `usr/local/include`) for the compiler and linker to find.
   - **Backend libs** should be installed in `/opt`
   - Extract it to your desired location:
    ```bash
    # install the frontend part (Can skip for Rust)
    tar xzvf icicle30-ubuntu22.tar.gz
    cp -r ./icicle/lib/* /usr/lib/
    cp -r ./icicle/include/icicle/ /usr/local/include/ # copy C++ headers
    # extract CUDA backend (OPTIONAL)
    tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /opt
     ```

    :::note
    Installing the frontend is optional for Rust. Rust does not use it.    
    :::

    :::tip
    You may install to any directory, but you need to ensure it can be found by the linker at compile and runtime.
    You can install anywhere and use a symlink to ensure it can be easily found as if it were in the default directory.
    :::

2. **Linking Your Application**:

  Applications need to link to the ICICLE device library and to every field and/or curve library. The backend libraries are dynamically loaded at runtime, so there is no need to link to them.

  **C++**
   - When compiling your C++ application, link against the ICICLE libraries:
     ```bash
     g++ -o myapp myapp.cpp -licicle_device -licicle_field_bn254 -licicle_curve_bn254
     # if not installed in standard dirs, for example /custom/path/, need to specify it
     g++ -o myapp myapp.cpp -I/custom/path/icicle/include -L/custom/path/icicle/lib -licicle_device -licicle_field_bn254 -licicle_curve_bn254 -Wl,-rpath,/custom/path/icicle/lib/
     ```

   - Or via cmake
    ```bash
    # Add the executable
    add_executable(example example.cpp)
    # Link the libraries
    target_link_libraries(example icicle_device icicle_field_bn254 icicle_curve_bn254)

    # OPTIONAL (if not installed in default location)

    # The following is setting compile and runtime paths for headers and libs assuming
    #   - headers in /custom/path/icicle/include
    #   - libs in/custom/path/icicle/lib

    # Include directories
    target_include_directories(example PUBLIC /custom/path/icicle/include)
    # Library directories
    target_link_directories(example PUBLIC /custom/path/icicle/lib/)
    # Set the RPATH so linker finds icicle libs at runtime
    set_target_properties(example PROPERTIES
                          BUILD_RPATH /custom/path/icicle/lib/
                          INSTALL_RPATH /custom/path/icicle/lib/)
    ```

  :::tip
  If you face linkage issues, try `ldd myapp` to see the runtime dependencies. If ICICLE libs are not found, you need to add the install directory to the search path of the linker. In a development environment, you can do that using the environment variable export `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/custom/path/icicle/lib` or similar (for non-Linux). For deployment, make sure it can be found and avoid using LD_LIBRARY_PATH.

  Alternatively, you can embed the search path in the app as an rpath by adding `-Wl,-rpath,/custom/path/icicle/lib/`. This is demonstrated above.
  :::

  **Rust**
     - When building the ICICLE crates, ICICLE frontend libs are built from source, along with the Rust bindings. They are installed to `target/<buildtype>/deps/icicle`, and Cargo will link them correctly. Note that you still need to install the CUDA backend if you have a CUDA GPU.
     - Simply use `cargo build` or `cargo run` and it should link to ICICLE libs.

  **Go** - TODO

:::warning
When deploying an application (whether in C++, Rust, or Go), you must make sure to either deploy the ICICLE libs (that you download or build from source) along with the application binaries (as tar, Docker image, package manager installer, or otherwise) or make sure to install ICICLE (and the backend) on the target machine. Otherwise, the target machine will have linkage issues.
:::

## Backend Loading

The ICICLE library dynamically loads backend libraries at runtime. By default, it searches for backends in the following order:

1. **Environment Variable**: If the `ICICLE_BACKEND_INSTALL_DIR` environment variable is defined, ICICLE will prioritize this location.
2. **Default Directory**: If the environment variable is not set, Icicle will search in the default directory `/opt/icicle/lib/backend`.

:::warning
If building ICICLE frontend from source, make sure to load a backend that is compatible with the frontend version. CUDA backend libs are forward compatible with newer frontends (e.g., CUDA-backend-3.0 works with ICICLE-3.2). The opposite is not guaranteed.
:::

If you install in a custom dir, make sure to set `ICICLE_BACKEND_INSTALL_DIR`:
```bash
ICICLE_BACKEND_INSTALL_DIR=path/to/icicle/lib/backend/ myapp # for an executable myapp
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
func LoadBackendFromEnvOrDefault() EIcicleError
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
func LoadBackend(path string, isRecursive bool) EIcicleError
```
- **`path`**: The directory where the backend libraries are located.
- **`isRecursive`**: If `true`, the function will search for backend libraries recursively within the specified path.
