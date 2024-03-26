cmake_minimum_required(VERSION 3.18)

include(Common.cmake)

set_env()
set_gpu_env()

project(icicle LANGUAGES CUDA CXX)

option(BUILD_TESTS "Build tests" OFF)

if((DEFINED CURVE) AND (DEFINED FIELD))
  message( FATAL_ERROR "CURVE and FIELD cannot be defined at the same time" )
endif ()

if(DEFINED FIELD)
  add_subdirectory(src/fields)
endif ()

if(DEFINED CURVE)
  set(FIELD ${CURVE})
  add_subdirectory(src/fields)
  add_subdirectory(src/curves)
endif ()

option(BUILD_HASH "Build hash functions" ON)
if(BUILD_HASH)
  add_subdirectory(src/hash)
endif ()