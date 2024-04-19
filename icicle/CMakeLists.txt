cmake_minimum_required(VERSION 3.18)

include(cmake/Common.cmake)
include(cmake/FieldsCommon.cmake)
include(cmake/CurvesCommon.cmake)

set_env()
set_gpu_env()

project(icicle LANGUAGES CUDA CXX)

option(DEVMODE "Enable development mode" OFF)
if (DEVMODE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 --ptxas-options=-O0 --ptxas-options=-allow-expensive-optimizations=false -DDEVMODE=ON")
endif ()

if((DEFINED CURVE) AND (DEFINED FIELD))
  message( FATAL_ERROR "CURVE and FIELD cannot be defined at the same time" )
endif ()

option(EXT_FIELD "Build extension field" OFF)
if(DEFINED FIELD)
  check_field()
  add_subdirectory(src/fields)
endif ()

option(G2 "Build G2" OFF)
option(ECNTT "Build ECNTT" OFF)
if(DEFINED CURVE)
  check_curve()
  set(FIELD ${CURVE})
  add_subdirectory(src/fields)
  add_subdirectory(src/curves)
endif ()

option(BUILD_HASH "Build hash functions" OFF)
if(BUILD_HASH)
  add_subdirectory(src/hash)
endif ()