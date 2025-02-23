cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(example)
find_package(Vulkan REQUIRED)
find_library(LOG_LIB log REQUIRED)  # Android log library

add_executable(example example.cpp)
target_include_directories(example PRIVATE "../../../icicle/include" "..")
target_link_libraries(example PRIVATE Vulkan::Vulkan)
target_link_directories(example PRIVATE "${CMAKE_SOURCE_DIR}/build/icicle")
message("${CMAKE_BINARY_DIR}/icicle")
target_link_libraries(example PRIVATE  icicle_field_babybear icicle_device ${LOG_LIB})
if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()
