cmake_minimum_required(VERSION 3.18)

project(icicle_fe)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Add the include directory
include_directories(include)

option(BUILD_TESTS "Build unit tests" OFF)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
