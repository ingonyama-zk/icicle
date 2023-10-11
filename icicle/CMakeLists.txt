cmake_minimum_required(VERSION 3.18)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# add the target cuda architectures
# each additional architecture increases the compilation time and output file size
if(${CMAKE_VERSION} VERSION_LESS "3.24.0")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
  set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
endif()

project(icicle LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

option(BUILD_TESTS "Build tests" OFF)

if(NOT BUILD_TESTS)
  add_library(
    icicle STATIC
    curves/bn254/lde.cu
    curves/bn254/msm.cu
    curves/bn254/projective.cu
    curves/bn254/ve_mod_mult.cu
    curves/bls12_377/lde.cu
    curves/bls12_377/msm.cu
    curves/bls12_377/projective.cu
    curves/bls12_377/ve_mod_mult.cu
    curves/bls12_381/lde.cu
    curves/bls12_381/msm.cu
    curves/bls12_381/projective.cu
    curves/bls12_381/ve_mod_mult.cu
    curves/bls12_381/poseidon.cu
  )
  
  set_target_properties(icicle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(icicle PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(icicle PROPERTIES OUTPUT_NAME "ingo_icicle")
  target_compile_options(icicle PRIVATE -c)

else()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
  )

  # For Windows: Prevent overriding the parent project's compiler/linker settings

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(
    primitives_test
    primitives/test.cu
  )
  target_link_libraries(
    primitives_test
    GTest::gtest_main
  )

  include(GoogleTest)
  set_target_properties(primitives_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  gtest_discover_tests(primitives_test)
endif()