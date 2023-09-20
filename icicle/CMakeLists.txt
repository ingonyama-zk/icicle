cmake_minimum_required(VERSION 3.18)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# add the target cuda architectures
# each additional architecture increases the compilation time and output file size
if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
endif ()
project(icicle LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")


# when adding a new curve, add the respective item here
# curves need to be addressed by numbers cos string comparison of preprocessor insturctions is unavailable
set(SUPPORTED_CURVES 254;12381;12377)

set(IS_CURVE_SUPPORTED FALSE)
foreach (SUPPORTED_CURVE ${SUPPORTED_CURVES})
  if (CURVE STREQUAL SUPPORTED_CURVE)
    set(IS_CURVE_SUPPORTED TRUE)
  endif ()
endforeach()

if (NOT IS_CURVE_SUPPORTED)
  message( FATAL_ERROR "The value of CURVE variable: ${CURVE} is not one of the supported curves: ${SUPPORTED_CURVES}" )
endif ()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCURVE=${CURVE}")

option(BUILD_TESTS "Build tests" OFF)

if (NOT BUILD_TESTS)

  add_library(
    icicle
    appUtils/msm/msm.cu
  )
  target_compile_options(icicle PRIVATE -c)

  add_custom_command(
    TARGET icicle
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} ARGS --redefine-sym msm_cuda=${CURVE_NAME}_msm_cuda --redefine-sym msm_internal_cuda=${CURVE_NAME}_msm_internal_cuda ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/msm/msm.cu.o
  )
  add_custom_command(
    TARGET icicle
    POST_BUILD
    COMMAND ${CMAKE_AR} ARGS -rcs ${LIBRARY_OUTPUT_DIRECTORY}/libingo_${CURVE_NAME}.a ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/msm/msm.cu.o
  )
  file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/msm/msm.cu.o)

else()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings

  # boosting lib
  include_directories("/home/miner/include/boost_1_80_0")

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

endif ()
