cmake_minimum_required(VERSION 3.18)

project(icicle_fe)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_CPU_BE "Build CPU backend" ON) # TODO Yuval remove from here once seperating the backends from frontend
option(BUILD_CUDA_BE "Build CUDA backend" ON) # TODO Yuval remove from here once seperating the backends from frontend

# device API library
add_library(icicle_device STATIC 
src/device_api.cpp 
src/errors.cpp
)

# Add the include directory
target_include_directories(icicle_device PUBLIC include)
include_directories(include)

# field library
add_library(icicle_field STATIC 
src/vec_ops/vec_ops.cpp
)


if (BUILD_CPU_BE) 
  add_subdirectory(backend/cpu)
endif()

if (BUILD_CUDA_BE) 
  add_subdirectory(backend/cuda)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
