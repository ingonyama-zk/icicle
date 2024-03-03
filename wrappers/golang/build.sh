#!/bin/bash

G2_DEFINED=OFF

if [[ $2 ]]
then
  G2_DEFINED=ON
fi

BUILD_DIR=$(realpath "$PWD/../../icicle/build")
SUPPORTED_CURVES=("bn254" "bls12_377" "bls12_381" "bw6_761")

if [[ $1 == "all" ]]
then
  BUILD_CURVES=("${SUPPORTED_CURVES[@]}")
else
  BUILD_CURVES=( $1 )
fi

cd ../../icicle
mkdir -p build

for CURVE in "${BUILD_CURVES[@]}"
do
  cmake -DCURVE=$CURVE -DG2_DEFINED=$G2_DEFINED -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build
done