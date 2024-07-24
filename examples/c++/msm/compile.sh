#!/bin/bash

set -e

mkdir -p build/example
mkdir -p build/icicle

# Build Icicle
cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DG2=ON -S ../../../icicle_v3/ -B build/icicle
cmake --build build/icicle -j

# Configure and build the example application
cmake -S . -B build/example
cmake --build build/example