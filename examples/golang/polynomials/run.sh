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
mkdir -p build/icicle
BUILD_DIR=$(realpath "build/icicle")
ICICLE_DIR=$(realpath "../../../icicle/")
ICICLE_BACKEND_SOURCE_DIR="${ICICLE_DIR}/backend/${DEVICE_TYPE_LOWERCASE}"

# Build Icicle and the example app that links to it
if [ "$DEVICE_TYPE" != "CPU" ] && [ ! -d "${ICICLE_BACKEND_INSTALL_DIR}" ] && [ -d "${ICICLE_BACKEND_SOURCE_DIR}" ]; then
  echo "Building icicle and ${DEVICE_TYPE} backend"
  rm -f "${BUILD_DIR}/CMakeCache.txt"
  cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DECNTT=OFF "-D${DEVICE_TYPE}_BACKEND"=local -S "${ICICLE_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j
  rm "${BUILD_DIR}/CMakeCache.txt"
  cmake -DCMAKE_BUILD_TYPE=Release -DFIELD=babybear -DECNTT=OFF "-D${DEVICE_TYPE}_BACKEND"=local -S "${ICICLE_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "${BUILD_DIR}/backend")
else
  echo "Building icicle without backend, ICICLE_BACKEND_INSTALL_DIR=${ICICLE_BACKEND_INSTALL_DIR}"
  export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR}"
  rm -f "${BUILD_DIR}/CMakeCache.txt"
  cmake -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -S "${ICICLE_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j
  rm "${BUILD_DIR}/CMakeCache.txt"
  cmake -DCMAKE_BUILD_TYPE=Release -DFIELD=babybear -S "${ICICLE_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j
fi

export CGO_LDFLAGS="-L${BUILD_DIR} -lstdc++ -Wl,-rpath,${BUILD_DIR}"

go run main.go -device "${DEVICE_TYPE}"
