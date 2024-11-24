#!/bin/bash

METAL_BACKEND=${1:-main}
BACKEND_DIR=${2:-./icicle/backend}

# Check if BACKEND_DIR exists
if [ ! -d "${BACKEND_DIR}" ]; then
    echo "Error: Directory '${BACKEND_DIR}' does not exist."
    exit 1
fi

# Get the absolute path of the backend directory
ABS_METAL_DIR=$(realpath ${BACKEND_DIR})/metal

echo "Trying to pull Metal backend commit '${METAL_BACKEND}' to '${ABS_METAL_DIR}'"

exit 1

if [ -d "${ABS_METAL_DIR}" ] && [ "$(ls -A ${ABS_METAL_DIR})" ]; then
    echo "Directory ${ABS_METAL_DIR} is not empty."
    read -p "Do you want to proceed with fetching and resetting? (y/n): " response
    case "$response" in
        [Yy]* )
            echo "Proceeding with fetch and reset..."
            cd ${ABS_METAL_DIR}
            git fetch origin
            git reset --hard origin/${METAL_BACKEND}
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
    echo "Directory ${ABS_METAL_DIR} is empty or does not exist. Cloning..."
    mkdir -p ${ABS_METAL_DIR}
    cd ${ABS_METAL_DIR}
    git clone git@github.com:ingonyama-zk/icicle-metal-backend.git ${ABS_METAL_DIR}
    git checkout ${METAL_BACKEND}
fi