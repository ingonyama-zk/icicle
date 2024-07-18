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
option(CPU_BACKEND "Build CPU backend" ON)
option(CUDA_BACKEND "Branch/commit to pull for CUDA backend, local path or OFF to disable pulling" OFF)

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

if (CPU_BACKEND)
  add_subdirectory(backend/cpu)
endif()

if (CUDA_BACKEND)
  string(TOLOWER "${CUDA_BACKEND}" CUDA_BACKEND_LOWER)
  if (CUDA_BACKEND_LOWER STREQUAL "local")
    # CUDA backend is local, no need to pull    
    message(STATUS "Adding CUDA backend from local path: icicle/backend/cuda")
    add_subdirectory(backend/cuda)

    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend")
  else()
    set(CUDA_BACKEND_URL "git@github.com:ingonyama-zk/icicle-cuda-backend.git")

    include(FetchContent)
    message("-- Fetching cuda backend from ${CUDA_BACKEND_URL}:${CUDA_BACKEND}")
    FetchContent_Declare(
      cuda_backend
      GIT_REPOSITORY ${CUDA_BACKEND_URL}
      GIT_TAG ${CUDA_BACKEND}
    )
    FetchContent_MakeAvailable(cuda_backend)
    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/_deps/cuda_backend-build")
    endif()
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

