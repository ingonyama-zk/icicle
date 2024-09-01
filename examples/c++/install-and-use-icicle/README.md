# Example: Install and use ICICLE (C++)

This example demonstrates how to install ICICLE binaries and use them in a C++ application.

Download release binaries from our [github release page](https://github.com/ingonyama-zk/icicle/releases):
- **Frontend** icicle30-ubuntu22.tar.gz
- **Backend** icicle30-ubuntu22-cuda122.tar.gz

> [!NOTE]
> The names of the files are based on the release version. Ensure you update the tar file names in the example if you’re using a different release.

## Optional: Using Docker

While not mandatory, this example can be demonstrated in an Ubuntu 22 Docker container.
```bash
docker run -it --rm --gpus all -v ./:/workspace -w /workspace icicle-release-ubuntu22-cuda122 bash
```

This command starts a bash session in the Docker container, with GPUs enabled and the example files mapped to /workspace in the container.

### Building the docker image

The Docker image is based on NVIDIA’s image for Ubuntu 22.04 and can be built from the following Dockerfile:

```dockerfile
# Use the official NVIDIA development runtime image for Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    tar
```

Build the Docker image with the following command:
```bash
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .`
```

## Extract tars and install ICICLE

Extracting and Installing the Frontend
```bash
cd release
# extract frontend part
tar xzvf icicle30-ubuntu22.tar.gz
cp -r ./icicle/lib/* /usr/lib/
cp -r ./icicle/include/icicle/ /usr/local/include/ # copy C++ headers
```

Extracting and Installing the CUDA Backend (Optional)

```bash
# extract CUDA backend (OPTIONAL)
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /opt
rm -rf icicle # remove the extracted dir
```

## Compile and Link the C++ Example with ICICLE

```bash
cd ..
mkdir build
cmake -S . -B build && cmake --build build
```

## Launch the executable

```bash
./build/example
```

## Install ICICLE in a Custom Location

If installing in a custom location such as /custom/path:
```bash
mkdir -p /custom/path
cd release
tar xzvf icicle30-ubuntu22.tar.gz -C /custom/path
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /custom/path # OPTIONAL
```

### Build your app and link to ICICLE

When installing ICICLE in a custom location, you need to specify the paths for the include and library directories so that the compiler, linker, and loader can find them during compile and runtime. Add the following to your CMake file:
```cmake
# Include directories
target_include_directories(example PUBLIC /custom/path/icicle/include)
# Library directories
target_link_directories(example PUBLIC /custom/path/icicle/lib/)
# Set the RPATH so linker finds icicle libs at runtime
set_target_properties(example PROPERTIES
                      BUILD_RPATH /custom/path/icicle/lib/
                      INSTALL_RPATH /custom/path/icicle/lib/)
```

Compile and Launch the Executable

```bash
cd ..
mkdir build
cmake -S . -B build && cmake --build build
```

### Launch the executable

Since CUDA backend is installed to `/custom/path` we need to set the env variable accordingly:
```bash
export ICICLE_BACKEND_INSTALL_DIR=/custom/path/icicle/lib/backend
./build/example
```

Alternatively, you can use the following API in your code:
```cpp
extern "C" eIcicleError icicle_load_backend(const char* path, bool is_recursive);
```
