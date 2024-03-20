cmake_minimum_required(VERSION 3.18)

include(../Common.cmake)

set_env()
set_gpu_env()

project(icicle LANGUAGES CUDA CXX)

# when adding a new curve/field, append its name to the end of this list
set(SUPPORTED_CURVES bn254;bls12_381;bls12_377;bw6_761;grumpkin)
SET(SUPPORTED_CURVES_WITHOUT_NTT grumpkin)

set(IS_CURVE_SUPPORTED FALSE)
set(I 0)
foreach (SUPPORTED_CURVE ${SUPPORTED_CURVES})
  math(EXPR I "${I} + 1")
  if (CURVE STREQUAL SUPPORTED_CURVE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCURVE_ID=${I}")
    set(IS_CURVE_SUPPORTED TRUE)
  endif ()
endforeach()

if (NOT IS_CURVE_SUPPORTED)
  message( FATAL_ERROR "The value of CURVE variable: ${CURVE} is not one of the supported curves: ${SUPPORTED_CURVES}" )
endif ()

if (G2_DEFINED STREQUAL "ON")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DG2_DEFINED=ON")
endif ()

option(BUILD_TESTS "Build tests" OFF)

if (NOT BUILD_TESTS)

  message(STATUS "Building without tests.")

  add_library(
    icicle
    utils/vec_ops.cu
    utils/mont.cu
    primitives/projective.cu
    appUtils/msm/msm.cu
    ${ICICLE_SOURCES}
  )
  set_target_properties(icicle PROPERTIES OUTPUT_NAME "ingo_${CURVE}")
  target_compile_definitions(icicle PRIVATE CURVE=${CURVE})  

else()

  message(STATUS "Building tests.")

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
    runner
    tests/runner.cu
  )

  target_link_libraries(
    runner
    GTest::gtest_main
  )

  include(GoogleTest)
  set_target_properties(runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  gtest_discover_tests(runner)

endif ()
