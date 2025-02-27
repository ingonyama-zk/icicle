#!/bin/bash



ICICLE_PATH='/Users/lnm-at-work/dev-setups/stas-android-vulcan-issue/icicle'
rm -r ./build
rm -r ${ICICLE_PATH}/build

docker run -it -v $ICICLE_PATH:/workspace/icicle  toochain-android-vulkan # todo: plug in path to icicle base via arg