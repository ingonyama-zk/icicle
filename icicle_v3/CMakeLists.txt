cmake_minimum_required(VERSION 3.18)

project(icicle_fe)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/field.cmake)
include(cmake/curve.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_CPU_BE "Build CPU backend" ON) # TODO Yuval remove from here once separating the backends from frontend
option(BUILD_CUDA_BE "Build CUDA backend" ON) # TODO Yuval remove from here once separating the backends from frontend

# device API library
add_library(icicle_device SHARED
  src/device_api.cpp
  src/errors.cpp
  src/runtime.cpp
)
target_link_libraries(icicle_device PUBLIC dl)
target_include_directories(icicle_device PUBLIC include)

if((DEFINED CURVE) AND (DEFINED FIELD))
  message( FATAL_ERROR "CURVE and FIELD cannot be defined at the same time" )
endif ()

# field library
if(FIELD)
  check_field()
  setup_field_target()
endif()

if(CURVE)
  check_curve()
  setup_curve_target()
endif ()


if (BUILD_CPU_BE)
  add_subdirectory(backend/cpu)
endif()

if (BUILD_CUDA_BE)
  add_subdirectory(backend/cuda)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
