#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

for dir in $(find $(dirname "$0")/$1 -mindepth 1 -maxdepth 1 -type d); do
  dir_name=$(basename "$dir")
  if [ -d "$dir" ] && [ "$dir_name" != "target" ] && [ "$dir_name" != "build" ] && [ "$dir_name" != "out" ]; then
    echo "================================================================"
    echo "======== Running $dir example ========="
    echo "================================================================"
    pushd $dir
    ./run.sh -d CPU
    popd
    echo "================================================================"
    echo "======== Finished $dir example ========="
    echo "================================================================"
    echo ""
  fi
done
