

# Install and use ICICLE

## Optional: use a docker for env with install permissions if need it.
```bash
docker run -it --rm --gpus all -v ./:/workspace -w /workspace icicle-release-ubuntu22-cuda122 bash
```

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
# extract CUDA backend
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /opt
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
