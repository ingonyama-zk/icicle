#!/bin/bash

set -e 

# Build dockers

# Ubuntu 22.04, CUDA 12.2.2
docker build -t icicle-release-ubuntu22-cuda122 -f Dockerfile.ubuntu22 .
# Ubuntu 20.04, CUDA 12.2.2
docker build -t icicle-release-ubuntu20-cuda122 -f Dockerfile.ubuntu20 .
# ubi7 (rhel compatible), CUDA 12.2.2
docker build -t icicle-release-ubi7-cuda122 -f Dockerfile.ubi7 .
# ubi8 (rhel compatible), CUDA 12.2.2
docker build -t icicle-release-ubi8-cuda122 -f Dockerfile.ubi8 .
# ubi7 (rhel compatible), CUDA 12.2.2
docker build -t icicle-release-ubi9-cuda122 -f Dockerfile.ubi9 .

# compile and tar release in each

mkdir -p release_output && rm -rf release_output/* # output dir where tars will be placed

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

# ubi 7
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v ./release_output:/output     \
            -v ./scripts:/scripts           \
            icicle-release-ubi7-cuda122 bash /scripts/release/build_release_and_tar.sh icicle30 ubi7 cuda122

# ubi 8
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v ./release_output:/output     \
            -v ./scripts:/scripts           \
            icicle-release-ubi8-cuda122 bash /scripts/release/build_release_and_tar.sh icicle30 ubi8 cuda122

# ubi 9
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v ./release_output:/output     \
            -v ./scripts:/scripts           \
            icicle-release-ubi9-cuda122 bash /scripts/release/build_release_and_tar.sh icicle30 ubi9 cuda122

