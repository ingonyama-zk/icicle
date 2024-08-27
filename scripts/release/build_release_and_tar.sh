#!/bin/bash

set -e

# List of fields and curves
fields=("babybear" "stark252")
curves=("bn254" "bls12_381" "bls12_377" "bw6_761" "grumpkin")

cd /
mkdir install_dir && mkdir install_dir/icicle # output dir that is tared

# Iterate over fields
for field in "${fields[@]}"; do
    echo "Building for field: $field"

    mkdir build -p && rm -rf build/*
    # Configure, build, and install
    cmake -S icicle -B build -DFIELD=$field -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=install_dir/icicle
    cmake --build build -j  # build
    cmake --install build   # install
done

# Iterate over curves
for curve in "${curves[@]}"; do
    echo "Building for curve: $curve"

    mkdir build -p && rm -rf build/*
    # Configure, build, and install
    cmake -S icicle -B build -DCURVE=$curve -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=install_dir/icicle
    cmake --build build -j  # build
    cmake --install build   # install
done

# Create the tarball
cd install_dir
tar -czvf /output/${OUTPUT_TAR_NAME} icicle # tar the install dir
