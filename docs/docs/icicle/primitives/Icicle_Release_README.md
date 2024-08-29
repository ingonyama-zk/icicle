
# Icicle Release README

## Overview

Icicle is a powerful C++ library designed to provide flexible and efficient computation through its modular backend architecture. This README explains how to build and release Icicle for multiple Linux distributions, including Ubuntu 20.04, Ubuntu 22.04, and CentOS 7. It also describes the content of a release and how to use the generated tar files.

## Content of a Release

Each Icicle release includes a tar file containing the build artifacts for a specific distribution. The tar file includes the following structure:

- **`./icicle/include/`**: This directory contains all the necessary header files for using the Icicle library from C++.
  
- **`./icicle/lib/`**: 
  - **Icicle Libraries**: All the core Icicle libraries are located in this directory. Applications linking to Icicle will use these libraries.
  - **Backends**: The `./icicle/lib/backend/` directory houses backend libraries, including the CUDA backend. While the CUDA backend is included, it will only be used on machines with a GPU. On machines without a GPU, the CUDA backend is not utilized.

### Considerations

Currently, the CUDA backend is included in every installation tar file, even on machines without a GPU. This ensures consistency across installations but results in additional files being installed that may not be used. 

**TODO**: Consider splitting the release into two separate tar files—one with the CUDA backend and one without—depending on the target machine’s hardware capabilities.

## Build Docker Image

To build the Docker images for each distribution and CUDA version, use the following commands:

```bash
# Ubuntu 22.04, CUDA 12.2
docker build -t icicle-release-ubuntu22-cuda122 -f Dockerfile.ubuntu22 .

# Ubuntu 20.04, CUDA 12.2
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .

# CentOS 7, CUDA 12.2
docker build -t icicle-release-centos7-cuda122 -f Dockerfile.centos7 .
```

### Docker Environment Explanation

The Docker images you build represent the target environment for the release. Each Docker image is tailored to a specific distribution and CUDA version. You first build the Docker image, which sets up the environment, and then use this Docker image to build the release tar file. This ensures that the build process is consistent and reproducible across different environments.

## Build Libraries Inside the Docker

To build the Icicle libraries inside a Docker container and output the tar file to the `release_output` directory:

```bash
mkdir -p release_output
docker run --rm --gpus all              -v ./icicle:/icicle                 -v ./release_output:/output         -v ./scripts:/scripts               icicle-release-ubuntu22-cuda122 bash /scripts/release/build_release_and_tar.sh
```

This command executes the `build_release_and_tar.sh` script inside the Docker container, which provides the build environment. It maps the source code and output directory to the container, ensuring the generated tar file is available on the host system.

You can replace `icicle-release-ubuntu22-cuda122` with another Docker image tag to build in the corresponding environment (e.g., Ubuntu 20.04 or CentOS 7).

## Installing and Using the Release

1. **Extract the Tar File**:
   - Download the appropriate tar file for your distribution (Ubuntu 20.04, Ubuntu 22.04, or CentOS 7).
   - Extract it to your desired location:
     ```bash
     tar -xzvf icicle-<distribution>-cuda122.tar.gz -C /path/to/install/location
     ```

2. **Linking Your Application**:
   - When compiling your C++ application, link against the Icicle libraries found in `./icicle/lib/`:
     ```bash
     g++ -o myapp myapp.cpp -L/path/to/icicle/lib -licicle_device -licicle_field_or_curve
     ```
   - Note: You only need to link to the Icicle device and field or curve libraries. The backend libraries are dynamically loaded at runtime.

## Backend Loading

The Icicle library dynamically loads backend libraries at runtime. By default, it searches for backends in the following order:

1. **Environment Variable**: If the `ICICLE_BACKEND_INSTALL_DIR` environment variable is defined, Icicle will prioritize this location.
2. **Default Directory**: If the environment variable is not set, Icicle will search in the default directory `/opt/icicle/lib/backend`.

### Custom Backend Loading

If you need to load a backend from a custom location at any point during runtime, you can call the following function:

```cpp
extern "C" eIcicleError icicle_load_backend(const char* path, bool is_recursive);
```

- **`path`**: The directory where the backend libraries are located.
- **`is_recursive`**: If `true`, the function will search for backend libraries recursively within the specified path.

---

