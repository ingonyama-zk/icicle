#!/bin/bash

# Exit immediately if a command exits with a non-zero status
rm -r build
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
: "${ICICLE_BACKEND_INSTALL_DIR:=../../../icicle/backend/vulkan}"

DEVICE_TYPE_LOWERCASE=$(echo "$DEVICE_TYPE" | tr '[:upper:]' '[:lower:]')

echo -e "\033[33m$DEVICE_TYPE\033[0m"
# Create necessary directories
mkdir -p build/example
mkdir -p build/icicle

ICICLE_DIR=$(realpath "../../../icicle/")
echo -e "\033[33m$ICICLE_DIR\033[0m"
ICICLE_BACKEND_SOURCE_DIR="${ICICLE_DIR}/backend/${DEVICE_TYPE_LOWERCASE}"
echo -e "\033[33m$ICICLE_BACKEND_SOURCE_DIR\033[0m"
echo -e  "\033[33m$NDK_DIR\033[0m"
# Build Icicle and the example app that links to it
if [ "$DEVICE_TYPE" != "CPU" ] && [ ! -d "${ICICLE_BACKEND_INSTALL_DIR}" ] && [ -d "${ICICLE_BACKEND_SOURCE_DIR}" ]; then
  echo -e "\033[33mConfigure icicle and ${DEVICE_TYPE} backend\033[0m"
  cmake -DVULKAN_BACKEND=local -DBUILD_TESTS=OFF -DBUILD_FOR_ANDROID=ON -DANDROID_PLATFORM=android-35 -DCMAKE_BUILD_TYPE=Release -DFIELD=babybear -DEXT_FIELD=OFF -DG2=OFF -DECNTT=OFF -S "${ICICLE_DIR}" -B build/icicle
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "build/icicle/backend")
  echo -e "\033[33mCompleted configuring icicle and ${DEVICE_TYPE} backend\033[0m"
else
  echo "Building icicle without backend, ICICLE_BACKEND_INSTALL_DIR=${ICICLE_BACKEND_INSTALL_DIR}"
  export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR}"
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_FOR_ANDROID=ON  -DFIELD=babybear -DEXT_FIELD=OFF -S "${ICICLE_DIR}" -B build/icicle
fi
echo -e "\033[33mConfigure example\033[0m"
cmake -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r28/build/cmake/android.toolchain.cmake -DANDROID_NDK=/opt/android-ndk-r28 -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-35 -DCMAKE_BUILD_TYPE=Release -S . -B build/example # todo use variable for install dir
echo -e "\033[33mCompleted configuring example\033[0m"

echo -e "\033[33mBuilding icicle and ${DEVICE_TYPE} backend\033[0m"
cmake --build build/icicle -j
echo -e "\033[33mCompleted building icicle and ${DEVICE_TYPE} backend\033[0m"
cmake --build build/example -j
echo -e "\033[33mCompleted building example\033[0m"

echo -e "\033[33mSyncing dependencies over Android Debug Bridge\033[0m"

# ${adb} push -p ./build /data/local/tmp/
adbsync  -q --exclude './icicle/_deps/**' push ./build /data/local/tmp


echo -e "\033[33mRunning example over Android Debug Bridge\033[0m"

# clear logcat messages
adb logcat -c

#  redirect logcat ICICLE message to ADB shell

# ${adb} shell logcat -v time  "ICICLE:*" "*:S"  & 

# run example

# adb shell logcat -v time  "ICICLE:*" "*:S"  &  adb shell "LD_LIBRARY_PATH=/data/local/tmp/build/icicle:/data/local/tmp/build/icicle/backend/vulkan ICICLE_BACKEND_INSTALL_DIR=/data/local/tmp/build/icicle/backend /data/local/tmp/build/example/example VULKAN"
adb shell "LD_LIBRARY_PATH=/data/local/tmp/build/icicle:/data/local/tmp/build/icicle/backend/vulkan ICICLE_BACKEND_INSTALL_DIR=/data/local/tmp/build/icicle/backend /data/local/tmp/build/example/example VULKAN"

# sleep 5
# adb shell pkill logcat

