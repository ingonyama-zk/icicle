cmake_minimum_required(VERSION 3.18)

project(icicle_cpu_backend)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO Yuval: handle incdir when backend is seperated from frontend
# TODO Yuval: dll?

# device API library
add_library(icicle_cpu_device STATIC cpu_device_api.cpp)
target_link_libraries(icicle_cpu_device PUBLIC icicle_device)

# field API library
add_library(icicle_cpu_field STATIC field/cpu_vec_ops.cpp)
target_link_libraries(icicle_cpu_field PUBLIC icicle_device icicle_field)
