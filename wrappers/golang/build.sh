#!/bin/bash

G2_DEFINED=OFF
ECNTT_DEFINED=OFF
CUDA_COMPILER_PATH=/usr/local/cuda/bin/nvcc
DEVMODE=OFF

SUPPORTED_CURVES=("bn254" "bls12_377" "bls12_381" "bw6_761")

# Parse arguments
# Handle curve
if [[ $1 == "all" ]]
then
  BUILD_CURVES=("${SUPPORTED_CURVES[@]}")
else
  BUILD_CURVES=( $1 )
fi

shift # skip curve argument
for arg in "$@"
do
    arg_lower=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
    case "$arg_lower" in
        -cuda_version=*)
            cuda_version=$(echo "$arg" | cut -d'=' -f2)
            CUDA_COMPILER_PATH=/usr/local/cuda-$cuda_version/bin/nvcc
            ;;
        -ecntt*)
            # ECNTT_DEFINED=$(echo "$arg" | cut -d'=' -f2)
            ECNTT_DEFINED=ON
            ;;
        -g2*)
            # G2_DEFINED=$(echo "$arg" | cut -d'=' -f2)
            G2_DEFINED=ON
            ;;
        -devmode*)
            # DEVMODE=$(echo "$arg" | cut -d'=' -f2)
            DEVMODE=ON
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

BUILD_DIR=$(realpath "$PWD/../../icicle/build")

cd ../../icicle
mkdir -p build

for CURVE in "${BUILD_CURVES[@]}"
do
  echo "CURVE=${CURVE}" > build_config.txt
  echo "ECNTT=${ECNTT_DEFINED}" >> build_config.txt
  echo "G2=${G2_DEFINED}" >> build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DCURVE=$CURVE -DG2=$G2_DEFINED -DECNTT=$ECNTT_DEFINED -DDEVMODE=$DEVMODE -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build -j8 && rm build_config.txt
done