#!/bin/bash

# Exit immediately on error
set -e

mkdir -p build/example
mkdir -p build/icicle

# Configure and build Icicle
cmake -S ../../../icicle/ -B build/icicle -DMSM=OFF -DCMAKE_BUILD_TYPE=Debug -DCURVE=bn254
cmake --build build/icicle -j

# Configure and build the example application
cmake -DCMAKE_BUILD_TYPE=Debug -S . -B build/example
cmake --build build/example
