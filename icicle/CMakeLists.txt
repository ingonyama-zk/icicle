cmake_minimum_required(VERSION 3.18)

project(icicle LANGUAGES CUDA CXX)

include(cmake/Common.cmake)
include(cmake/FieldsCommon.cmake)
include(cmake/CurvesCommon.cmake)

set_env()
set_gpu_env()

option(DEVMODE "Enable development mode" OFF)
option(EXT_FIELD "Build extension field" OFF)
option(G2 "Build G2" OFF)
option(ECNTT "Build ECNTT" OFF)
option(BUILD_HASH "Build hash functions" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
# add options here

if((DEFINED CURVE) AND (DEFINED FIELD))
  message( FATAL_ERROR "CURVE and FIELD cannot be defined at the same time" )
endif ()

if (DEVMODE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 --ptxas-options=-O0 --ptxas-options=-allow-expensive-optimizations=false -DDEVMODE=ON")
endif ()

if(DEFINED FIELD)
  check_field()
  add_subdirectory(src/fields)
endif ()

if(DEFINED CURVE)
  check_curve()
  set(FIELD ${CURVE})
  add_subdirectory(src/fields)
  add_subdirectory(src/curves)
endif ()

if (G2)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DG2")
endif ()

if (EXT_FIELD)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DEXT_FIELD")
endif ()

if(BUILD_HASH)
  add_subdirectory(src/hash)
endif ()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
