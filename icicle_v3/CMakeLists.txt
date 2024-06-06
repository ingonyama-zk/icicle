cmake_minimum_required(VERSION 3.18)

project(icicle_v3)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/field.cmake)
include(cmake/curve.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NTT "Build NTT" ON)
option(MSM "Build MSM" ON)
option(EXT_FIELD "Build extension field" OFF)
option(ECNTT "Build ECNTT" OFF)
option(G2 "Build G2" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_CPU_BE "Build CPU backend" ON) 
option(BUILD_CUDA_BE "Build CUDA backend" ON) # TODO Yuval remove from here once separating the cuda backend

# device API library
add_library(icicle_device SHARED
  src/device_api.cpp
  src/errors.cpp
  src/runtime.cpp
)
target_link_libraries(icicle_device PUBLIC dl)
target_include_directories(icicle_device PUBLIC include)

if((DEFINED CURVE) AND (DEFINED FIELD))
  if(NOT ("${CURVE}" STREQUAL "${FIELD}"))
    message(FATAL_ERROR "CURVE and FIELD should be defined at the same time unless they are equal")
  endif()
endif()

# curve is building the scalar field too
if(CURVE)
  check_curve()
  setup_curve_target()
elseif(FIELD)
  check_field()
  setup_field_target()
endif()

if (BUILD_CPU_BE)
  add_subdirectory(backend/cpu)
endif()

if (BUILD_CUDA_BE)
  add_subdirectory(backend/cuda)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
