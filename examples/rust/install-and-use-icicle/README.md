

# Example: Install and use ICICLE

This example shows how to install CUDA backend and use it in Rust application.

Download release binaries from our [github release page](https://github.com/ingonyama-zk/icicle/releases):
- **Backend** icicle30-ubuntu22-cuda122.tar.gz

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

RUN apt install cargo -y
```

Build the Docker image with the following command:
```bash
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .
```

## Extract and Install the CUDA Backend

```bash
cd release
# extract CUDA backend
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /opt
```

## Build the Rust Application and Execute

Add the following dependencies to your Cargo.toml file:

```cargo
[dependencies]
icicle-runtime = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
icicle-core = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
icicle-babybear = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
```

Build and Run the Application

```bash
cargo run --release
```

## Install in a Custom Location
If you prefer to install the CUDA backend in a custom location such as /custom/path, follow these steps:

```bash
mkdir -p /custom/path
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /custom/path
```

If installed in a custom location, you need to define the environment variable:

```bash
export ICICLE_BACKEND_INSTALL_DIR=/custom/path/icicle/lib/backend
```

Alternatively, you can load the backend programmatically in your Rust code using:
```bash
pub fn load_backend(path: &str) -> Result<(), eIcicleError>
```
