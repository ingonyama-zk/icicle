## Build the release

This section is describing how a release is generated, given the release sources.<br>
We use docker to represent the target environment for the release. Each Docker image is tailored to a specific distribution and CUDA version. You first build the Docker image, which sets up the environment, and then use this Docker image to build the release tar file. This ensures that the build process is consistent and reproducible across different environments.

### Build Docker Image

The Docker images represent the target environment for the release.

To build the Docker images for each distribution and CUDA version, use the following commands:

```bash
# Ubuntu 22.04, CUDA 12.2
docker build -t icicle-release-ubuntu22-cuda122 -f Dockerfile.ubuntu22 .

# Ubuntu 20.04, CUDA 12.2
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .

# CentOS 7, CUDA 12.2
docker build -t icicle-release-centos7-cuda122 -f Dockerfile.centos7 .
```


## Build Libraries Inside the Docker

To build the Icicle libraries inside a Docker container and output the tar file to the `release_output` directory:

```bash
mkdir -p release_output
# ubuntu 22
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v ./release_output:/output     \
            -v ./scripts:/scripts           \
            icicle-release-ubuntu22-cuda122 bash /scripts/release/build_release_and_tar.sh icicle30 ubuntu22 cuda122

# ubuntu 20
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v ./release_output:/output     \
            -v ./scripts:/scripts           \
            icicle-release-ubuntu20-cuda122 bash /scripts/release/build_release_and_tar.sh icicle30 ubuntu20 cuda122
```

This command executes the `build_release_and_tar.sh` script inside the Docker container, which provides the build environment. It maps the source code and output directory to the container, ensuring the generated tar file is available on the host system.

You can replace `icicle-release-ubuntu22-cuda122` with another Docker image tag to build in the corresponding environment (e.g., Ubuntu 20.04 or CentOS 7).
Make sure to pass corresponding OS and CUDA version in the params `icicle30 ubuntu22 cuda122`. For example for centos7 it would be `icicle30 centos7 cuda122`.

