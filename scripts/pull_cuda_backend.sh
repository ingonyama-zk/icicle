#!/bin/bash

CUDA_BACKEND=${1:-main}
BACKEND_DIR=${2:-./icicle/backend}

# Check if BACKEND_DIR exists
if [ ! -d "${BACKEND_DIR}" ]; then
    echo "Error: Directory '${BACKEND_DIR}' does not exist."
    exit 1
fi
# Get the absolute path of the backend directory
ABS_CUDA_DIR=$(realpath ${BACKEND_DIR})/cuda

echo "Trying to pull CUDA backend commit '${CUDA_BACKEND}' to '${ABS_CUDA_DIR}'"

if [ -d "${ABS_CUDA_DIR}" ] && [ "$(ls -A ${ABS_CUDA_DIR})" ]; then
    echo "Directory ${ABS_CUDA_DIR} is not empty."
    read -p "Do you want to proceed with fetching and resetting? (y/n): " response
    case "$response" in
        [Yy]* )
            echo "Proceeding with fetch and reset..."
            cd ${ABS_CUDA_DIR}
            git fetch origin
            git reset --hard origin/${CUDA_BACKEND}
            ;;
        [Nn]* )
            echo "Aborting."
            exit 1
            ;;
        * )
            echo "Invalid input. Aborting."
            exit 1
            ;;
    esac
else
    echo "Directory ${ABS_CUDA_DIR} is empty or does not exist. Cloning..."
    mkdir -p ${ABS_CUDA_DIR}
    cd ${ABS_CUDA_DIR}
    git clone git@github.com:ingonyama-zk/icicle-cuda-backend.git ${ABS_CUDA_DIR}
    git checkout ${CUDA_BACKEND}
fi