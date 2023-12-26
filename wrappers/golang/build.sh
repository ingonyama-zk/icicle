#!/bin/bash

SUDO=''
if [ "$EUID" != 0 ]; then 
  echo "Icicle setup script should be run with root privileges, please run this as root"
  SUDO='sudo'
fi

# Build icicle normally with cmake
# take an argument for which curve to use for -DCURVE
# check that the curve is supported
# 
# cd to ../../icicle
# mkdir -p build
# cmake -DCURVE=$1 -S . -B build
# cmake --build build
# copy to pwd

SUPPORT_CURVES=( "bn254" "bls12_377" "bls12_381" "bw6_761" )

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$BUILD_DIR
