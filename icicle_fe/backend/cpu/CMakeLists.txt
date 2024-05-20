cmake_minimum_required(VERSION 3.18)

project(icicle_cpu_backend)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO Yuval: handle incdir when backend is seperated from frontend

# TODO Yuval: dll?
add_library(icicle_cpu_backend SHARED cpu_device_api.cpp)
