cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(example)

add_executable(example example.cpp)
target_include_directories(example PRIVATE "../../../icicle_v3/include" ".." "../../../icicle_v3")
target_link_directories(example PRIVATE "${CMAKE_SOURCE_DIR}/build/icicle")
message("${CMAKE_BINARY_DIR}/icicle")
target_link_libraries(example PRIVATE icicle_curve_bn254 icicle_field_bn254 icicle_device)
if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()

