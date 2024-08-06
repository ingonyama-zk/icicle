pushd /Users/koren/Documents/REPOS/icicle/icicle_v3
rm -rf build/*
cmake -DCURVE=bn254 -DBUILD_CUDA_BE=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -S . -B build
cmake --build build -j
mkdir ./build/generated_data
popd
