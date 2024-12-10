#!/bin/bash

set -e 

# Check if sufficient arguments are provided
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <version> [output_dir]"
  echo
  echo "Arguments:"
  echo "  <version>       The version for release tar files (required)."
  echo "  [output_dir]    The directory to store release output files (optional, defaults to './release_output')."
  exit 1
fi

# Use provided release_output directory or default to "release_output"
version="$1"
output_dir="${2:-./release_output}"

first_char=${version:0:1}
if [[ "${first_char,,}" == "v" ]]; then
    version="${version:1}"
fi

version="${version//./_}"

# Check if both directories exist in the current working directory
if [[ ! -d "./icicle" || ! -d "./scripts" ]]; then
  echo "Usage: The current directory must contain both 'icicle' and 'scripts' directories. Retry from icicle root dir."
  exit 1
fi

# Build Docker images
echo "Building Docker images..."
# Ubuntu 22.04, CUDA 12.2.2
docker build -t icicle-release-ubuntu22-cuda122 -f ./scripts/release/Dockerfile.ubuntu22 .
# Ubuntu 20.04, CUDA 12.2.2
docker build -t icicle-release-ubuntu20-cuda122 -f ./scripts/release/Dockerfile.ubuntu20 .
# ubi8 (rhel compatible), CUDA 12.2.2
docker build -t icicle-release-ubi8-cuda122 -f ./scripts/release/Dockerfile.ubi8 .
# ubi9 (rhel compatible), CUDA 12.2.2
docker build -t icicle-release-ubi9-cuda122 -f ./scripts/release/Dockerfile.ubi9 .

# Compile and tar release in each

# Inform the user of what is being done
echo "Preparing release files..."
echo "Version: $version"
echo "Output Directory: $output_dir"

# Create the output directory if it doesn't exist, and clean it
mkdir -p "$output_dir" && rm -rf "$output_dir/*"

# ubuntu 22
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v "$output_dir:/output"        \
            -v ./scripts:/scripts           \
            icicle-release-ubuntu22-cuda122 bash /scripts/release/build_release_and_tar.sh icicle_$version ubuntu22 cuda122 &

# ubuntu 20
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v "$output_dir:/output"        \
            -v ./scripts:/scripts           \
            icicle-release-ubuntu20-cuda122 bash /scripts/release/build_release_and_tar.sh icicle_$version ubuntu20 cuda122 &

# ubi 8 (rhel compatible)
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v "$output_dir:/output"        \
            -v ./scripts:/scripts           \
            icicle-release-ubi8-cuda122 bash /scripts/release/build_release_and_tar.sh icicle_$version ubi8 cuda122 &

# ubi 9 (rhel compatible)
docker run --rm --gpus all                  \
            -v ./icicle:/icicle             \
            -v "$output_dir:/output"        \
            -v ./scripts:/scripts           \
            icicle-release-ubi9-cuda122 bash /scripts/release/build_release_and_tar.sh icicle_$version ubi9 cuda122 &
