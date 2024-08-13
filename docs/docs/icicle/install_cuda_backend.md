
# CUDA Backend

## Overview

The CUDA backend in ICICLE V3 is a high-performance, closed-source component designed to accelerate cryptographic computations using NVIDIA GPUs. This backend includes a set of specialized libraries optimized for different cryptographic fields and curves, providing significant speedups for operations such as MSM, NTT, and elliptic curve operations.

## Installation

### Default Installation Path

Typically, the CUDA backend is installed to the following directory:

```
/opt/icicle/backend/
``` 

### Custom Installation Path

If you choose to install the backend to a different location, you can specify the path using the `ICICLE_BACKEND_INSTALL_DIR` environment variable. ICICLE will attempt to load the backend from this specified path.

#### Symlink Option

Alternatively, you can create a symbolic link from the default installation directory to your custom install directory. This allows ICICLE to load the backend as if it were installed in the default location.

Example:

```bash
sudo ln -s /custom/path/to/icicle/backend /opt/icicle/backend
```

### Licensing

The CUDA backend requires a valid license to function. Licenses are available for purchase [here](#). After purchasing, you will receive a license key that must be installed in the backend directory.

For licensing instructions and detailed information, refer to the licensing documentation provided with your purchase or contact our support team for assistance.

TODO update section and the link in license part above.

## Usage

### Loading the Backend

You load the CUDA backend using the provided C++/Rust/Go API functions.

#### C++

```cpp
/**
 * @brief Attempts to load the backend from either the environment variable or the default install directory.
 *
 * This function first checks if the environment variable `ICICLE_BACKEND_INSTALL_DIR` is set and points to an existing
 * directory. If so, it attempts to load the backend from that directory. If the environment variable is not set or the
 * directory does not exist, it falls back to the default directory (`/opt/icicle/backend`). If neither option is
 * successful, the function returns an error.
 *
 * @return eIcicleError The status of the backend loading operation, indicating success or failure.
 */
extern "C" eIcicleError icicle_load_backend_from_env_or_default();

/**
 * @brief Load ICICLE backend into the process from the specified install directory.
 *
 * @param path Path of the backend library or directory where backend libraries are installed.
 * @return eIcicleError Status of the loaded backend.
 */
extern "C" eIcicleError icicle_load_backend(const char* path, bool is_recursive);
```

These functions provide flexibility in how you load the backend, either by specifying a custom path or by relying on environment variables and default paths.

#### Rust

```
pub fn load_backend_from_env_or_default() -> Result<(), eIcicleError>;
pub fn load_backend(path: &str) -> Result<(), eIcicleError>;
```

#### Go

```
TODO
```

---

For more detailed instructions on installing and using the CUDA backend, refer to the [full documentation](#) or contact our support team for assistance.