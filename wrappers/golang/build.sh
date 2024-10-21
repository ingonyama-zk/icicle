#!/bin/bash

MSM_DEFINED=ON
NTT_DEFINED=ON
G2_DEFINED=ON
ECNTT_DEFINED=ON
EXT_FIELD=ON
HASH=ON

CUDA_COMPILER_PATH=/usr/local/cuda/bin/nvcc

DEVMODE=OFF
BUILD_CURVES=( )
BUILD_FIELDS=( )

SUPPORTED_CURVES=("bn254" "bls12_377" "bls12_381" "bw6_761", "grumpkin")
SUPPORTED_FIELDS=("babybear")
CUDA_BACKEND=OFF

BUILD_DIR="${ICICLE_BUILD_DIR:-$(realpath "$PWD/../../icicle/build")}"

if [[ $1 == "-help" ]]; then
  echo "Build script for building ICICLE cpp libraries"
  echo ""
  echo "If more than one curve or more than one field is supplied, the last one supplied will be built"
  echo ""
  echo "USAGE: ./build.sh [OPTION...]"
  echo ""
  echo "OPTIONS:"
  echo "  -curve=<curve_name>       Specifies the curve to be built. If \"all\" is supplied,"
  echo "                            all curves will be built with any additional curve options."
  echo ""
  echo "  -skip_msm                 Builds the curve library with MSM (multi-scalar multiplication) disabled."
  echo ""
  echo "  -skip_ntt                 Builds the curve/field library with NTT (number theoretic transform) disabled."
  echo ""
  echo "  -skip_g2                  Builds the curve library with G2 (a secondary group) disabled."
  echo ""
  echo "  -skip_ecntt               Builds the curve library with ECNTT (elliptic curve NTT) disabled."
  echo ""
  echo "  -skip_hash                Builds the library with Hashes (Keccak, Sha3, etc) disabled."
  echo ""
  echo "  -field=<field_name>       Specifies the field to be built. If \"all\" is supplied,"
  echo "                            all fields will be built with any additional field options."
  echo ""
  echo "  -skip_fieldext            Builds the field library with the extension field disabled."
  echo ""
  echo "  -cuda_backend=<option>    Specifies the branch/commit to pull for CUDA backend, or \"local\" if it's"
  echo "                            located under icicle/backend/cuda."
  echo "                            Default: \"OFF\""
  echo ""
  echo "  -cuda_version=<version>   Specifies the version of CUDA to use for compilation."
  echo ""
  exit 0
fi

for arg in "$@"
do
    arg_lower=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
    case "$arg_lower" in
        -cuda_version=*)
            cuda_version=$(echo "$arg" | cut -d'=' -f2)
            CUDA_COMPILER_PATH=/usr/local/cuda-$cuda_version/bin/nvcc
            ;;
        -cuda_backend=*)
            CUDA_BACKEND=$(echo "$arg_lower" | cut -d'=' -f2)
            ;;
        -skip_msm)
            MSM_DEFINED=OFF
            ;;
        -skip_ntt)
            NTT_DEFINED=OFF
            ;;
        -skip_ecntt)
            ECNTT_DEFINED=OFF
            ;;
        -skip_g2)
            G2_DEFINED=OFF
            ;;
        -skip_hash)
            HASH=OFF
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
        -skip_fieldext)
            EXT_FIELD=OFF
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

cd ../../icicle
mkdir -p build
rm -f "$BUILD_DIR/CMakeCache.txt"

for CURVE in "${BUILD_CURVES[@]}"
do
  echo "CURVE=${CURVE}" > build_config.txt
  echo "MSM=${MSM_DEFINED}" >> build_config.txt
  echo "NTT=${NTT_DEFINED}" >> build_config.txt
  echo "ECNTT=${ECNTT_DEFINED}" >> build_config.txt
  echo "G2=${G2_DEFINED}" >> build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DCUDA_BACKEND=$CUDA_BACKEND -DCURVE=$CURVE -DMSM=$MSM_DEFINED -DNTT=$NTT_DEFINED -DG2=$G2_DEFINED -DECNTT=$ECNTT_DEFINED -DHASH=OFF -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build --target install -j8 && rm build_config.txt
done

# Needs to remove the CMakeCache.txt file to allow building fields after curves
# have been built since CURVE and FIELD cannot both be defined
rm -f "$BUILD_DIR/CMakeCache.txt"

for FIELD in "${BUILD_FIELDS[@]}"
do
  echo "FIELD=${FIELD}" > build_config.txt
  echo "NTT=${NTT_DEFINED}" >> build_config.txt
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  echo "EXT_FIELD=${EXT_FIELD}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DCUDA_BACKEND=$CUDA_BACKEND -DFIELD=$FIELD -DNTT=$NTT_DEFINED -DEXT_FIELD=$EXT_FIELD -DHASH=OFF -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build --target install -j8 && rm build_config.txt
done

if [[ $HASH == "ON" ]]; then
  echo "DEVMODE=${DEVMODE}" >> build_config.txt
  cmake -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER_PATH -DCUDA_BACKEND=$CUDA_BACKEND -DHASH=$HASH -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build --target install -j8 && rm build_config.txt
fi
