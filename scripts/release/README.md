## Build the release

This section is describing how a release is generated, given the release sources.<br>
We use docker to represent the target environment for the release. Each Docker image is tailored to a specific distribution and CUDA version. You first build the Docker image, which sets up the environment, and then use this Docker image to build the release tar file. This ensures that the build process is consistent and reproducible across different environments.

## Build full release

To build all tars:
```bash
# from icicle root dir
mkdir -p release_output && rm -rf release_output/* # output dir where tars will be placed
./scripts/release/build_all.sh release_output # release_output is the output dir where tar files will be generated to
```

### Build Docker Image

The Docker images represent the target environment for the release.

To build the Docker images for each distribution and CUDA version, use the following commands:

```bash
cd ./scripts/release
# Ubuntu 22.04, CUDA 12.2.2
docker build -t icicle-release-ubuntu22-cuda122 -f Dockerfile.ubuntu22 .
```


## Build Libraries Inside the Docker

To build the Icicle libraries inside a Docker container and output the tar file to the `release_output` directory:

```bash
# from icicle root dir
mkdir -p release_output
# ubuntu 22
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v ./release_output:/output     \
            -v ./scripts:/scripts           \
            icicle-release-ubuntu22-cuda122 bash /scripts/release/build_release_and_tar.sh icicle30 ubuntu22 cuda122          
```

This command executes the `build_release_and_tar.sh` script inside the Docker container, which provides the build environment. It maps the source code and output directory to the container, ensuring the generated tar file is available on the host system.

You can replace `icicle-release-ubuntu22-cuda122` with another Docker image tag to build in the corresponding environment.
Make sure to pass corresponding OS and CUDA version in the params `icicle30 ubuntu22 cuda122`. For example for ubi9 it would be `icicle30 ubi9 cuda122`.
See `build_all.sh` script for reference.
