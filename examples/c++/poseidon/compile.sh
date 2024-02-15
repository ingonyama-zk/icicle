#!/bin/bash

# Exit immediately on error
set -e

rm -rf build
mkdir -p build
cmake -S . -B build
cmake --build build
