rm -rf build_vulkan/*
mkdir -p build_vulkan
cmake -S icicle/ -B build_vulkan/ -DCURVE=bn254 -DBUILD_TESTS=ON -DVULKAN_BACKEND=local -DCMAKE_BUILD_TYPE=Debug -DEXT_FIELD=OFF
cmake --build build_vulkan -j;
./build_vulkan/tests/test_field_api --gtest_filter="*NTTTest*" |& tee ntt_vulk_test.log





