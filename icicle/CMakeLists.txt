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


# when adding a new curve, append its name to the end of this list
set(SUPPORTED_CURVES bn254;bls12_381;bls12_377)

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

option(BUILD_TESTS "Build tests" OFF)

if (NOT BUILD_TESTS)

  add_library(
    icicle
    utils/error_handler.cu
    utils/utils_kernels.cu
    primitives/projective.cu
    appUtils/msm/msm.cu
    appUtils/ntt/ntt.cu
    appUtils/lde/lde.cu
  )
  # set_target_properties(icicle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # set_target_properties(icicle PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # # cuda_add_library(cudanet STATIC ${cu})
  # target_link_libraries(icicle ${CUDA_LIBRARIES})
  # target_link_libraries(icicle cudart cuda cublas curand)
  # message(STATUS "include & link cuda")
  # link_directories(/usr/local/cuda/lib64)
  target_compile_options(icicle PRIVATE -c)
  # find_package(CUDA QUIET REQUIRED)

  add_custom_command(
    TARGET icicle
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} ARGS --redefine-sym MSMCuda=${CURVE}MSMCuda ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/msm/msm.cu.o
    # COMMAND ${CMAKE_OBJCOPY} ARGS --prefix-symbols=${CURVE}_ ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/ntt/ntt.cu.o
    # COMMAND ${CMAKE_OBJCOPY} ARGS --prefix-symbols=${CURVE}_ ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/lde/lde.cu.o
    COMMAND ${CMAKE_AR} ARGS -rcs ${LIBRARY_OUTPUT_DIRECTORY}/libingo_${CURVE}.a 
    ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/utils/error_handler.cu.o
    ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/primitives/projective.cu.o
    ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/utils/utils_kernels.cu.o 
    ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/msm/msm.cu.o 
    ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/ntt/ntt.cu.o
    ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/lde/lde.cu.o
                                                                                 
  )
  # file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/msm/msm.cu.o)
  # file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/ntt/ntt.cu.o)
  # file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/appUtils/lde/lde.cu.o)
  # file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/primitives/projective.cu.o)
  # file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/utils/utils_kernels.cu.o)
  # file(REMOVE ${PROJECT_BINARY_DIR}/CMakeFiles/icicle.dir/utils/error_handler.cu.o)

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

endif ()
