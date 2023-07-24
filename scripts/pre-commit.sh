#!/bin/sh

# Find CUDA and CPP files and apply clang-format 
find ./ -iname *.h -or -iname *.cuh -or -iname *.cu | xargs clang-format -i -style=file

# Run go fmt across all packages under goicicle
go list goicicle/... | xargs go fmt

# Run rust fmt and clippy (requires compilation)
cargo fmt --all
cargo clippy --no-deps --fix
