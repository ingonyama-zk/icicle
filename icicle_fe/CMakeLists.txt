cmake_minimum_required(VERSION 3.18)

project(icicle_fe)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_CPU_BE "Build CPU backend" ON) # TODO Yuval remove from here once seperating the backends from frontend

# Add the include directory
include_directories(include)

add_library(icicle_device STATIC 
  src/device_api.cpp 
  src/errors.cpp
)

if (BUILD_CPU_BE) 
  add_subdirectory(backend/cpu)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
