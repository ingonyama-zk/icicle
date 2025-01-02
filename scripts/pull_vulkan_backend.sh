#!/bin/bash

VULKAN_BACKEND=${1:-main}
BACKEND_DIR=${2:-./icicle/backend}

# Check if BACKEND_DIR exists
if [ ! -d "${BACKEND_DIR}" ]; then
    echo "Error: Directory '${BACKEND_DIR}' does not exist."
    exit 1
fi

# Get the absolute path of the backend directory
ABS_VULKAN_DIR=$(realpath ${BACKEND_DIR})/vulkan

echo "Trying to pull vulkan backend commit '${VULKAN_BACKEND}' to '${ABS_VULKAN_DIR}'"

if [ -d "${ABS_VULKAN_DIR}" ] && [ "$(ls -A ${ABS_VULKAN_DIR})" ]; then
    echo "Directory ${ABS_VULKAN_DIR} is not empty."
    read -p "Do you want to proceed with fetching and resetting? (y/n): " response
    case "$response" in
        [Yy]* )
            echo "Proceeding with fetch and reset..."
            cd ${ABS_VULKAN_DIR}
            git fetch origin
            git reset --hard origin/${VULKAN_BACKEND}
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
    echo "Directory ${ABS_VULKAN_DIR} is empty or does not exist. Cloning..."
    mkdir -p ${ABS_VULKAN_DIR}
    cd ${ABS_VULKAN_DIR}
    git clone git@github.com:ingonyama-zk/icicle-vulkan-backend.git ${ABS_VULKAN_DIR}
    git checkout ${VULKAN_BACKEND}
fi