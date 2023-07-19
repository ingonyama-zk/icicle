#!/bin/bash

TARGET_BN254="libbn254.so"
TARGET_BLS12_381="libbls12_381.so"
TARGET_BLS12_377="libbls12_377.so"

make $1

TARGET_BN254_PATH=$(find `pwd` -name $TARGET_BN254)
TARGET_BLS12_381_PATH=$(find `pwd` -name $TARGET_BLS12_381)
TARGET_BLS12_377_PATH=$(find `pwd` -name $TARGET_BLS12_377)


if [[ "$TARGET_BLS12_377_PATH" != "" ]]; then
    echo "BLS12_377 @ $TARGET_BLS12_377_PATH"
    export LD_LIBRARY_PATH=$TARGET_BLS12_377_PATH
fi

if [[ "$TARGET_BN254_PATH" != "" ]]; then
    echo "BN254 found @ $TARGET_BN254_PATH"
    export LD_LIBRARY_PATH=$TARGET_BN254_PATH
fi

if [[ "$TARGET_BLS12_381_PATH" != "" ]]; then
    echo "BN254_PATH @ $TARGET_BLS12_381_PATH"
    export LD_LIBRARY_PATH=$TARGET_BLS12_381_PATH
fi
