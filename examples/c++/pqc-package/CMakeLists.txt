cmake_minimum_required(VERSION 3.18)
project(icicle_pqc_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ICICLE PQC installation directory
set(ICICLE_PQC_INSTALL_DIR "" CACHE PATH "Path to ICICLE PQC installation directory")

if(NOT ICICLE_PQC_INSTALL_DIR)
    message(FATAL_ERROR "Please specify: cmake .. -DICICLE_PQC_INSTALL_DIR=/path/to/icicle/install")
endif()

# Find ICICLE package
find_package(icicle_pqc_package REQUIRED PATHS ${ICICLE_PQC_INSTALL_DIR})

# Create executable
add_executable(pqc_example pqc_example.cpp)

# Link ICICLE PQC package
target_link_libraries(pqc_example PRIVATE icicle::icicle_pqc_package)

get_target_property(interface_includes icicle::icicle_pqc_package INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Interface includes for icicle::icicle_pqc_package: ${interface_includes}")
