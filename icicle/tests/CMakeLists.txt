cmake_minimum_required(VERSION 3.18)

include(../cmake/Common.cmake)
include(../cmake/CurvesCommon.cmake)
include(../cmake/FieldsCommon.cmake)

set_env()
set_gpu_env()

project(icicle_benchmarks LANGUAGES CUDA CXX)

if(DEFINED CURVE)
  if(DEFINED FIELD)
    message( FATAL_ERROR "CURVE and FIELD cannot be defined at the same time" )
  endif ()
  check_curve()
elseif(DEFINED FIELD)
  check_field()
endif ()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DG2=ON")

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
    runner.cu
)

target_link_libraries(
    runner
    GTest::gtest_main
)

include(GoogleTest)
set_target_properties(runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

gtest_discover_tests(runner)
