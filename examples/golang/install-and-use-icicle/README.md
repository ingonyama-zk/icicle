# Example: Install and use ICICLE

This example shows how to install CUDA backend and use it in a Go application.

Download release binaries from our [github release page](https://github.com/ingonyama-zk/icicle/releases):

- **Backend** icicle_3_9_0-ubuntu22-cuda122.tar.gz
- **Frontend** icicle_3_9_0-ubuntu22.tar.gz

> [!NOTE]
> The names of the files are based on the release version. Ensure you update the tar file names in the example if you are using a different release.

## Optional: Using Docker with Ubuntu 22

While not mandatory, this example can be demonstrated in an Ubuntu 22 Docker container.

```bash
docker run -it --rm --gpus all -v ./:/workspace -w /workspace icicle-release-ubuntu22-cuda122 bash
```

This command starts a bash session in the Docker container with GPUs enabled and the example files mapped to /workspace in the container.

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

ENV GOLANG_VERSION 1.21.1
RUN curl -L https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz | tar -xz -C /usr/local
ENV PATH="/usr/local/go/bin:${PATH}"
```

Build the Docker image with the following command:

```bash
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .
```

## Extract and Install prebuilt binaries

```bash
cd release
# extract CUDA backend
tar xzvf icicle_3_9_0-ubuntu22-cuda122.tar.gz -C </path/to/install/backend>
tar xzvf icicle_3_9_0-ubuntu22.tar.gz -C </path/to/install>
```

## Add ICICLE dependency and export ICICLE lib variables

Update your go.mod to include ICICLE as a dependency.

Export `ICICLE_BACKEND_INSTALL_DIR` to point to where you extracted the **backend** libraries

```sh
export ICICLE_BACKEND_INSTALL_DIR=/path/to/install/backend
```

Alternatively, you can load the backend programmatically in your Go code using the `LoadBackend` function from the `runtime` package:

```go
func LoadBackend(path string, isRecursive bool) EIcicleError

runtime.LoadBackend("/custom/path/to/backend", true)
```

Export `CGO_LDFLAGS` to point to where you extracted the **frontend** libraries

```sh
export CGO_LDFLAGS="-L</path/to/install> -lstdc++ -Wl,-rpath,</path/to/install>"
```

You can now add specific packages when you need them in your application and load the backend

```go
import (
  runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
  core "github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
  bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
  bn254MSM "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
)

// This loads the CUDA backend that you extracted to /opt
runtime.LoadBackendFromEnvOrDefault()
```
