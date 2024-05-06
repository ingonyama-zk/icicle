#!/bin/bash

G2_DEFINED=OFF
ECNTT_DEFINED=OFF
CUDA_COMPILER_PATH=/usr/local/cuda/bin/nvcc
DEVMODE=OFF
EXT_FIELD=OFF
BUILD_CURVES=( )
BUILD_FIELDS=( )
BUILD_HASHES=( )

SUPPORTED_CURVES=("bn254" "bls12_377" "bls12_381" "bw6_761", "grumpkin")
SUPPORTED_FIELDS=("babybear")
SUPPORTED_HASHES=("keccak")

for arg in "$@"
do
    arg_lower=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
    case "$arg_lower" in
        -cuda_version=*)
            cuda_version=$(echo "$arg" | cut -d'=' -f2)
            CUDA_COMPILER_PATH=/usr/local/cuda-$cuda_version/bin/nvcc
            ;;
        -ecntt)
            ECNTT_DEFINED=ON
            ;;
        -g2)
            G2_DEFINED=ON
            ;;
        -curve=*)
            curve=$(echo "$arg_lower" | cut -d'=' -f2)
            if [[ $curve == "all" ]]
            then
              BUILD_CURVES=("${SUPPORTED_CURVES[@]}")
            else
              BUILD_CURVES=( $curve )
            fi
            ;;
        -field=*)
            field=$(echo "$arg_lower" | cut -d'=' -f2)
            if [[ $field == "all" ]]
            then
              BUILD_FIELDS=("${SUPPORTED_FIELDS[@]}")
            else
              BUILD_FIELDS=( $field )
            fi
            ;;
        -field-ext)
            EXT_FIELD=ON
            ;;
        -hash*)
            hash=$(echo "$arg_lower" | cut -d'=' -f2)
            if [[ $hash == "all" ]]
            then
              BUILD_HASHES=("${SUPPORTED_HASHES[@]}")
            else
              BUILD_HASHES=( $hash )
            fi
            ;;
        -devmode)
            DEVMODE=ON
            ;;
        -help)
            echo "Build script for building ICICLE cpp libraries"
            echo ""
            echo "If more than one curve or more than one field is supplied, the last one supplied will be built"
            echo ""
            echo "USAGE: ./build.sh [OPTION...]"
            echo ""
            echo "OPTIONS:"
            echo "  -curve=<curve_name>       The curve that should be built. If \"all\" is supplied,"
            echo "                            all curves will be built with any other supplied curve options"
            echo "  -g2                       Builds the curve lib with G2 enabled"
            echo "  -ecntt                    Builds the curve lib with ECNTT enabled"
            echo "  -field=<field_name>       The field that should be built. If \"all\" is supplied,"
            echo "                            all fields will be built with any other supplied field options"
            echo "  -field-ext                Builds the field lib with the extension field enabled"
            echo "  -hash=<hash>              The name of the hash to build or "all" to build all supported hashes"
            echo "  -devmode                  Enables devmode debugging and fast build times"
            echo "  -cuda_version=<version>   The version of cuda to use for compiling"
            echo ""
            exit 0
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
rm -f "$BUILD_DIR/CMakeCache.txt"

for CURVE in "${BUILD_CURVES[@]}"
do
  echo "CURVE=${CURVE}" > build_config.txt
  echo "ECNTT=${ECNTT_DEFINED}" >> build_config.txt
  echo "G2=${G2_DEFINED}" >> build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DCURVE=$CURVE -DG2=$G2_DEFINED -DECNTT=$ECNTT_DEFINED -DDEVMODE=$DEVMODE -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build -j8 && rm build_config.txt
done

# Needs to remove the CMakeCache.txt file to allow building fields after curves
# have been built since CURVE and FIELD cannot both be defined
rm -f "$BUILD_DIR/CMakeCache.txt"

for FIELD in "${BUILD_FIELDS[@]}"
do
  echo "FIELD=${FIELD}" > build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DFIELD=$FIELD -DEXT_FIELD=$EXT_FIELD -DDEVMODE=$DEVMODE -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build -j8 && rm build_config.txt
done

for HASH in "${BUILD_HASHES[@]}"
do
  echo "HASH=${HASH_DEFINED}" > build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DBUILD_HASH=$HASH -DDEVMODE=$DEVMODE -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build -j8 && rm build_config.txt
done
