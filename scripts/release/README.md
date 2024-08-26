
# Build docker image
```bash
docker build -t icicle-release-ubuntu22-cuda122 -f Dockerfile.ubuntu22 .
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .
docker build -t icicle-release-centos7-cuda122 -f Dockerfile.centos7 .
```

# Build libs inside the docker
To build inside the docker and ouptut the tar:
```bash
docker run --rm --gpus all          \
    -v ./icicle:/icicle             \
    -v ./release_objects:/output    \
    -v ./scripts:/scripts           \
    icicle-release-ubuntu22-cuda122 bash /scripts/release/build_release_and_tar.sh
```

replace `icicle-release-ubuntu22-cuda122` with another docker image tag to build in this environment instead.
