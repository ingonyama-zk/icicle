rm -R build
mkdir -p build
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -S . -B build
cmake --build build
./build/runner --gtest_brief=1