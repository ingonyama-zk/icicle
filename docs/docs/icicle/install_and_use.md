
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
   
  **C++**
   - When compiling your C++ application, link against the Icicle libraries found in `/opt/icicle/lib` or other location:
     ```bash
     g++ -o myapp myapp.cpp -L/opt/icicle/lib -licicle_field_babybear -licicle_curve_bn254
     ```

    
  :::note
  You need to link to the Icicle device library and in addition link to each field or curve libraries. The backend libraries are dynamically loaded at runtime, so not linking to them.
  :::

  **Rust**
     - When building the icicle crates, icicle frontend libs are built from source, in addition to the rust bindings. They are installed to `target/<buildtype>/deps/icile` and the crate is linked to that at runtime.
     - Need to install CUDA backend only, if tou have a CUDA GPU.
     - Note: can install and link to the installed libs instead of building them from source. This is currently not supported but will be in a future release.

  **Go** - TODO

:::warning when deploying an application (either C++, Rust or Go), you must make sure to either deploy the icicle libs (in Rust it's in `target/<buildtype>/deps/icile` or the preinstalled ones) along the application binaries (as tar, docker image, package manager installer or else) or make sure to install icicle (and the backend) on the target machine. Otherwise the target machine will have linkage issues.
:::

:::tip
If you face linkage issues, try `ldd myapp` to see the runtime deps. If ICICLE libs are not found in the filesystem, you need to add the install directory to the search path of the linker. In a development env You can do that by adding the install dir to `export LD_LIBRARY_PATH=/path/to/icicle/lib` or corresponding variables. For deployment, make sure it can be found and avoid `LD_LIBRARY_PATH`.
:::

## Backend Loading

The Icicle library dynamically loads backend libraries at runtime. By default, it searches for backends in the following order:

1. **Environment Variable**: If the `ICICLE_BACKEND_INSTALL_DIR` environment variable is defined, Icicle will prioritize this location.
2. **Default Directory**: If the environment variable is not set, Icicle will search in the default directory `/opt/icicle/lib/backend`.

:::warning
Make sure to load a backend that is compatible to the frontend version. CUDA backend libs are forward compatible with newer frontends (e.g. CUDA-backend-3.0 works with ICICLE-3.2). The opposite is not guranteed.
:::

To load backend from ICICLE_BACKEND_INSTALL_DIR or `/opt/icicle/lib/backend` in your application:

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
## Build the release

This section is describing how a release is generated, given the release sources.
We use docker to represent the target environment for the release. Each Docker image is tailored to a specific distribution and CUDA version. You first build the Docker image, which sets up the environment, and then use this Docker image to build the release tar file. This ensures that the build process is consistent and reproducible across different environments.
