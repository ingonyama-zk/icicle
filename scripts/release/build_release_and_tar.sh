#!/bin/bash

set -e

# Accept ICICLE_VERSION, ICICLE_OS, and ICICLE_CUDA_VERSION as inputs or use defaults
ICICLE_VERSION=${1:-icicle30}      # Default to "icicle30" if not set
ICICLE_OS=${2:-unknown_os}              # Default to "unknown_os" if not set
ICICLE_CUDA_VERSION=${3:-cuda_unknown} # Default to "cuda_unknown" if not set

# List of fields and curves
fields=("babybear" "stark252" "m31")
curves=("bn254" "bls12_381" "bls12_377" "bw6_761" "grumpkin")

cd /
mkdir -p install_dir/icicle # output dir that is tarred

# Iterate over fields
for field in "${fields[@]}"; do
    echo "Building for field: $field"

    mkdir -p build && rm -rf build/*
    # Configure, build, and install
    # Precompile SASS for modern architectures (Turing, Ampere, etc.) and include PTX fallback (?)
    cmake -S icicle -B build -DFIELD=$field -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=install_dir/icicle -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="75;80;86"
    cmake --build build -j  # build
    cmake --install build   # install
done

# Iterate over curves
for curve in "${curves[@]}"; do
    echo "Building for curve: $curve"

    mkdir -p build && rm -rf build/*
    # Configure, build, and install
    # Precompile SASS for modern architectures (Turing, Ampere, etc.) and include PTX fallback (?)
    cmake -S icicle -B build -DCURVE=$curve -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=install_dir/icicle -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="75;80;86"
    cmake --build build -j  # build
    cmake --install build   # install
done

# Split CUDA binaries to a separate directory to tar them separately
mkdir -p install_dir_cuda_only/icicle/lib/backend
mv install_dir/icicle/lib/backend/* install_dir_cuda_only/icicle/lib/backend

# Copy headers
cp -r ./icicle/include install_dir/icicle

# Create the tarball for frontend libraries
cd install_dir
tar -czvf /output/${ICICLE_VERSION}-${ICICLE_OS}.tar.gz icicle # tar the install dir

# Create tarball for CUDA backend
cd ../install_dir_cuda_only
tar -czvf /output/${ICICLE_VERSION}-${ICICLE_OS}-${ICICLE_CUDA_VERSION}.tar.gz icicle # tar the install dir