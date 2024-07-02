cmake_minimum_required(VERSION 3.18)

project(icicle_v3)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/field.cmake)
include(cmake/curve.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NTT "Build NTT" ON)
option(MSM "Build MSM" ON)
option(EXT_FIELD "Build extension field" OFF)
option(ECNTT "Build ECNTT" OFF)
option(G2 "Build G2" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_CPU_BE "Build CPU backend" ON)
option(BUILD_CUDA_BE "Build CUDA backend" ON) # TODO Yuval remove from here once separating the cuda backend

# device API library
add_library(icicle_device SHARED
  src/device_api.cpp
  src/runtime.cpp
  src/config_extension.cpp
)
target_link_libraries(icicle_device PUBLIC dl)
include_directories(include)

# Define the install directory (default is /usr/local)
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix")
endif()
message("-- CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Specify the installation rules
install(TARGETS icicle_device
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if((DEFINED CURVE) AND (DEFINED FIELD))
  if(NOT ("${CURVE}" STREQUAL "${FIELD}"))
    message(FATAL_ERROR "CURVE and FIELD should be defined at the same time unless they are equal")
  endif()
endif()

# curve is building the scalar field too
if(CURVE)
  check_curve()
  setup_curve_target()
elseif(FIELD)
  check_field()
  setup_field_target()
endif()

if (BUILD_CPU_BE)
  add_subdirectory(backend/cpu)
endif()

if (BUILD_CUDA_BE)
  add_subdirectory(backend/cuda)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

