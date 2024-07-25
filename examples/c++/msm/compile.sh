#!/bin/bash

set -e

mkdir -p build/example
mkdir -p build/icicle

ICILE_DIR=$(realpath "../../../icicle_v3/")
ICICLE_CUDA_BACKEND_DIR="${ICILE_DIR}/backend/cuda"

# Build Icicle and the example app that links to it
if [ -d "${ICICLE_CUDA_BACKEND_DIR}" ]; then
    echo "building icicle with CUDA backend"
    cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DG2=ON -DCUDA_BACKEND=local -S "${ICILE_DIR}" -B build/icicle
    cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/example -DBACKEND_DIR=$(realpath "build/icicle/backend")
else
    echo "building icicle without CUDA backend"
    cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -S "${ICILE_DIR}" -B build/icicle    
    cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/example
fi
cmake --build build/icicle -j
cmake --build build/example -j