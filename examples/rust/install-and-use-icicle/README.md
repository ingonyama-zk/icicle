

# Example: Install and use ICICLE

This example shows how to install CUDA backend and use it in Rust application.

Download release binaries for CUDA backend:
- **Backend** icicle30-ubuntu22-cuda122.tar.gz

:::note
Name of the files is based on the release version. Make sure to update the tar file names in the example if using different release.
:::

## Optional: This example is demonstrated in an ubuntu22 docker but this is not mandatory.
```bash
docker run -it --rm --gpus all -v ./:/workspace -w /workspace icicle-release-ubuntu22-cuda122 bash
```

This command is starting bash in the docker, with GPUs and mapping the example files to `/workspace` in the docker.

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

RUN apt install cargo -y
```

by `docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .`

## Extract tars and install
```bash
cd release
# extract CUDA backend (OPTIONAL)
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /opt
```

## Build application

Define ICICLE deps in cargo:
```cargo
[dependencies]
icicle-runtime = { git = "https://github.com/ingonyama-zk/icicle.git", branch="yshekel/V3" }
icicle-core = { git = "https://github.com/ingonyama-zk/icicle.git", branch="yshekel/V3" }
icicle-babybear = { git = "https://github.com/ingonyama-zk/icicle.git", branch="yshekel/V3" }
```

Then build
```bash
cargo build --release
```

## Launch the executable
```bash
cargo run --release
```

### CUDA license
If using CUDA backend, make sure to have a CUDA backend license:
- For license server, specify address: `export ICICLE_LICENSE_SERVER_ADDR=port@ip`.
- For local license, specify path to license: `export ICICLE_LICENSE_SERVER_ADDR=path/to/license`. (TODO rename env variable)

## Install in custom location

If installing in a custom location such as /custom/path:
```bash
mkdir -p /custom/path
tar xzvf icicle30-ubuntu22-cuda122.tar.gz -C /custom/path
```

define `ICICLE_BACKEND_INSTALL_DIR=/custom/path/icicle/lib/backend` or use `pub fn load_backend(path: &str) -> Result<(), eIcicleError>`