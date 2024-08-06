#!/bin/bash

CUDA_BACKEND=${1:-main}
BACKEND_DIR=${2:-backend/cuda}

if [ "$CUDA_BACKEND" != "OFF" ]; then
    # Get the absolute path of the backend directory
    ABS_BACKEND_DIR=$(realpath ${BACKEND_DIR})

    echo "Trying to pull CUDA backend commit '${CUDA_BACKEND}' to '${ABS_BACKEND_DIR}'"

    if [ -d "${BACKEND_DIR}" ] && [ "$(ls -A ${BACKEND_DIR})" ]; then
        echo "Directory ${BACKEND_DIR} is not empty. Fetching and resetting..."
        cd ${BACKEND_DIR}
        git fetch origin
        git reset --hard origin/${CUDA_BACKEND}
    else
        echo "Directory ${BACKEND_DIR} is empty or does not exist. Cloning..."
        mkdir -p ${BACKEND_DIR}
        cd ${BACKEND_DIR}
        git clone https://github.com/ingonyama-zk/icicle-cuda-backend.git ${ABS_BACKEND_DIR}
        git checkout ${CUDA_BACKEND}
    fi
fi