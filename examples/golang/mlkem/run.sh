#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create necessary directories
mkdir -p build/icicle

BUILD_DIR=$(realpath "build/icicle")
ICICLE_DIR=$(realpath "../../../icicle/")

# Build Icicle PQC
echo "Building icicle and backend"
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_PQC_BACKEND=ON -S "${ICICLE_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

export CGO_LDFLAGS="-L${BUILD_DIR} -lstdc++ -Wl,-rpath,${BUILD_DIR}"

go run main.go
