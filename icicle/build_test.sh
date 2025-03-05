#!/bin/sh

mkdir -p build && rm -rf build/*
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DFIELD=babybear -DEXT_FIELD=ON -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=$PWD/build/mylib -DSUMCHECK=OFF -S . -B build

# Build and install the project
cmake --build build --target install -j || { echo "Build failed"; exit 1; }

# Navigate to build/tests directory
cd build/tests/ || { echo "Directory build/tests/ not found"; exit 1; }

# Run ctest with the specified test filter
ctest -R "FieldTest.FriHashAPi" --verbose