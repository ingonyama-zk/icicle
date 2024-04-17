#!/bin/bash

# Exit immediately on error
set -e

mkdir -p build

# Configure and build Icicle
cmake -S ../../../icicle/ -B ../../../icicle/build -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254
cmake --build ../../../icicle/build

# Configure and build the example application
cmake -S . -B build/
cmake --build build/