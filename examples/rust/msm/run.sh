#!/bin/bash

set -e

# Function to display usage information
show_help() {
  echo "Usage: $0 <CPU|CUDA>"
  exit 0
}

# Check if the -h flag is provided or no arguments are provided
if [ "$1" == "-h" ]; then
  show_help
fi

DEVICE_TYPE=${1:-CUDA}
ICILE_DIR=$(realpath "../../../icicle_v3/")
ICICLE_CUDA_BACKEND_SRC_DIR="${ICILE_DIR}/backend/cuda"

# Check if DEVICE_TYPE is CUDA and if the CUDA backend directory exists
if [ "$DEVICE_TYPE" == "CUDA" ] && [ -d "${ICICLE_CUDA_BACKEND_SRC_DIR}" ]; then  
  echo "Loading CUDA backend from ${ICICLE_CUDA_BACKEND_SRC_DIR}"  
  export ICICLE_CUDA_BACKEND_DIR=$(realpath ./target/release/deps/icicle/lib/backend)
  cargo run --release --features=cuda
else
  echo "Falling back to CPU backend"
  # Load CPU backend (replace with actual command)
  cargo run --release
fi

