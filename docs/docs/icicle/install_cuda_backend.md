
# CUDA Backend

## Overview

The CUDA backend in ICICLE V3 is a high-performance, closed-source component designed to accelerate cryptographic computations using NVIDIA GPUs. This backend includes a set of specialized libraries optimized for different cryptographic fields and curves, providing significant speedups for operations such as MSM, NTT, and elliptic curve operations.

## Installation

The CUDA backend is a closed-source component that requires a license. To install the CUDA backend:

1. **Download the CUDA backend package** from the [ICICLE website](#). TODO fix link.

2. **Install to the default path:**
   ```bash
   sudo tar -xzf icicle-cuda-backend.tar.gz -C /opt/icicle/backend/
   ```

3. **Set up the environment variable if you installed it in a custom location:**
   ```bash
   export ICICLE_BACKEND_INSTALL_DIR=/custom/path/to/icicle/backend
   # OR symlink
   sudo ln -s /custom/path/to/icicle/backend /opt/icicle/backend
   ```

4. **Load the backend in your application:**
   ```cpp
   extern "C" eIcicleError icicle_load_backend_from_env_or_default();
   // OR
   extern "C" eIcicleError icicle_load_backend(const char* path, bool is_recursive);
   ```
   Rust:
    ```rust
    pub fn load_backend_from_env_or_default() -> Result<(), eIcicleError>;
    pub fn load_backend(path: &str) -> Result<(), eIcicleError>;
    ```
    Go:
    ```
    TODO
    ```

5. **Acquire a license key** from the [ICICLE website](#) and follow the provided instructions to activate it.



### Licensing (TODO fix link)

The CUDA backend requires a valid license to function. Licenses are available for purchase [here](#). After purchasing, you will receive a license key that must be installed. **Specify the license server address:**
```
export ICICLE_LICNSE_SERVER_PATH=port@ip
```

For licensing instructions and detailed information, refer to the licensing documentation provided with your purchase or contact our support team for assistance.

TODO update section and the link in license part above.


