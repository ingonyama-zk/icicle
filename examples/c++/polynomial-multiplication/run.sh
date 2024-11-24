#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
show_help() {
  echo "Usage: $0 [-d DEVICE_TYPE] [-b BACKEND_INSTALL_DIR]"
  echo
  echo "Options:"
  echo "  -d DEVICE_TYPE            Specify the device type (default: CPU)"
  echo "  -b BACKEND_INSTALL_DIR    Specify the backend installation directory (default: empty)"
  echo "  -h                        Show this help message"
  exit 0
}

# Parse command line options
while getopts ":d:b:h" opt; do
  case ${opt} in
    d )
      DEVICE_TYPE=$OPTARG
      ;;
    b )
      ICICLE_BACKEND_INSTALL_DIR="$(realpath ${OPTARG})"
      ;;
    h )
      show_help
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      show_help
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument" 1>&2
      show_help
      ;;
  esac
done

# Set default values if not provided
: "${DEVICE_TYPE:=CPU}"
: "${ICICLE_BACKEND_INSTALL_DIR:=}"

DEVICE_TYPE_LOWERCASE=$(echo "$DEVICE_TYPE" | tr '[:upper:]' '[:lower:]')

# Create necessary directories
mkdir -p build/example
mkdir -p build/icicle

ICILE_DIR=$(realpath "../../../icicle/")
ICICLE_BACKEND_SOURCE_DIR="${ICILE_DIR}/backend/${DEVICE_TYPE_LOWERCASE}"

# Build Icicle and the example app that links to it
if [ "$DEVICE_TYPE" != "CPU" ] && [ ! -d "${ICICLE_BACKEND_INSTALL_DIR}" ] && [ -d "${ICICLE_BACKEND_SOURCE_DIR}" ]; then
  echo "Building icicle and ${DEVICE_TYPE} backend"
  cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DMSM=OFF -DG2=OFF -DECNTT=OFF "-D${DEVICE_TYPE}_BACKEND"=local -S "${ICILE_DIR}" -B build/icicle
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "build/icicle/backend")
else
  echo "Building icicle without backend, ICICLE_BACKEND_INSTALL_DIR=${ICICLE_BACKEND_INSTALL_DIR}"
  export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR}"
  cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -S "${ICILE_DIR}" -B build/icicle
fi
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/example

cmake --build build/icicle -j
cmake --build build/example -j

./build/example/example "$DEVICE_TYPE"
