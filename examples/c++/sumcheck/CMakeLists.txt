cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(example)
add_executable(example example.cpp)
install(TARGETS example
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
target_include_directories(example PRIVATE "../../../icicle/include" "..")
target_link_directories(example PRIVATE "${CMAKE_SOURCE_DIR}/build/icicle")
target_link_libraries(example PRIVATE icicle_curve_bn254 icicle_field_bn254 icicle_device icicle_hash)
find_package(Threads REQUIRED)
target_link_libraries(example PRIVATE Threads::Threads)
if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()
