
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
   - Download (TODO link to latest release) the appropriate tar files for your distribution (Ubuntu 20.04, Ubuntu 22.04, or UBI 7,8,9 for RHEL compatible binaries).
   - **Frontend libs and headers** should be installed in default search paths (such as `/usr/local/lib` and `usr/local/include`) for the compiler and linker to find.
   - **Backend libs** should be installed in `/opt`
   - Extract it to your desired location:
     ```bash
     # install the frontend part (Can skip for Rust)     
     tar -xzvf icicle30-<distribution>.tar.gz -C /path/to/extract/
     cp -r /path/to/extract/icicle/include /usr/local/include/icicle # or any other
     cp -r /path/to/extract/icicle/lib /usr/local/lib # or any other
     # install CUDA backend (Required for all programming-languages that want to use CUDA backend)
     tar -xzvf icicle30-<distribution>-cuda122.tar.gz -C /opt/ # or other non-default install directory
     ```

    :::note
    You may install to any directory but need to make sure it can be found by the linker at compile and runtime.
    For example can extract the frontend to `/opt` too.
    ::: 

    :::tip
    You can install anywhere and use a link so that it can be easily found as if in the default directory.
    :::

1. **Linking Your Application**:

  Apps need to link to the ICICLE device library and to every field and/or curve libraries. The backend libraries are dynamically loaded at runtime, so not linking to them.

  **C++**
   - When compiling your C++ application, link against the Icicle libraries:
     ```bash
     g++ -o myapp myapp.cpp -licicle_device -licicle_field_bn254 -licicle_curve_bn254

     # if not installed in standard dirs, for example /opt, need to specify it
     g++ -o myapp myapp.cpp -I/opt/icicle/include -L/opt/icicle/lib -licicle_device -licicle_field_bn254 -licicle_curve_bn254 -Wl,-rpath,/opt/icicle/lib/
     ```

    - Or via cmake
    ```bash
    # Add the executable
    add_executable(example example.cpp)
    # Link the libraries
    target_link_libraries(example icicle_device icicle_field_bn254 icicle_curve_bn254)

    # OPTIONAL (if not installed in default location)
    # Include directories
    include_directories(/path/to/install/dir/icicle/include)
    # Library directories
    link_directories(/path/to/install/dir/icicle/lib/)    
    
    # Set the RPATH so linker finds icicle libs at runtime
    set_target_properties(example PROPERTIES
        BUILD_RPATH /path/to/install/dir/icicle/lib/
        INSTALL_RPATH /path/to/install/dir/icicle/lib/)
    ```

    :::tip
    If you face linkage issues, try `ldd myapp` to see the runtime deps. If ICICLE libs are not found, you need to add the install directory to the search path of the linker. In a development env you can do that using the env variable `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/icicle/lib` or similar (for non linux). For deployment, make sure it can be found and avoid `LD_LIBRARY_PATH`.

    Alternatively you can embed the search path on the app as an `rpath` by adding `-Wl,-rpath,/path/to/icicle/lib/`. This is what is demonstrated above.
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
If building ICICLE frontend from source, make sure to load a backend that is compatible to the frontend version. CUDA backend libs are forward compatible with newer frontends (e.g. CUDA-backend-3.0 works with ICICLE-3.2). The opposite is not guaranteed.
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
