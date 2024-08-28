

# Example: Install and use ICICLE

Download release binaries:
- **Frontend** icicle30-ubuntu22.tar.gz
- **Backend** icicle30-ubuntu22-cuda122.tar.gz

:::note
Name of the files is based on the release version. Make sure to update the tar file names in the example if using different release.
:::

## Optional: This example is demonstrated in an ubuntu22 docker but this is not mandatory.
```bash
docker run -it --rm --gpus all -v ./:/workspace -w /workspace icicle-release-ubuntu22-cuda122 bash
```

This command is starting bash in the docker, with GPUs and mapping the example files to `/worksapce` in the docker.

### Building the docker image
This image is based on nvidia's image for ubuntu22. built from the Dockerfile:
```dockerfile
# Use the official NVIDIA development runtime image for Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    tar
```

by `docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .`

## Extract tars and install
```bash
cd release
# extract frontend part
tar xzvf icicle30-ubuntu22.tar.gz
cp -r ./icicle/lib/* /usr/lib/
cp -r ./icicle/include/icicle/ /usr/local/include/ # copy C++ headers
# extract CUDA backend (OPTIONAL)
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /opt
rm -rf icicle # remove the extracted dir
```

## Compile and link C++ example to icicle
```bash
cd ..
mkdir build
cmake -S . -B build && cmake --build build
```

## Launch the executable
```bash
./build/example
```

### CUDA license
If using CUDA backend, make sure to have a CUDA backend license:
- For license server, specify address: `export ICICLE_LICENSE_SERVER_ADDR=port@ip`.
- For local license, specify path to license: `export ICICLE_LICENSE_SERVER_ADDR=path/to/license`. (TODO rename env variable)

## Install in custom location

If installing in a custom location such as /custom/path:
```bash
mkdir -p /custom/path
cd release
tar xzvf icicle30-ubuntu22.tar.gz -C /custom/path
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /custom/path # OPTIONAL
```

### Build your app and link to ICICLE
You will have to specify paths for include and libs so that the compiler linker and loader can find them at compile anb runtime.
You can add the following to cmake file to do so:
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

Alternatively, the example code can use the foolowing API instead:
```cpp
extern "C" eIcicleError icicle_load_backend(const char* path, bool is_recursive);
```