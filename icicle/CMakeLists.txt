cmake_minimum_required(VERSION 3.18)

project(icicle)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/field.cmake)
include(cmake/curve.cmake)
include(cmake/hash.cmake)

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

# Print the selected build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find the ccache program
find_program(CCACHE_PROGRAM ccache)
# If ccache is found, use it as the compiler launcher
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")

    # Use ccache for C and C++ compilers
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
else()
    message(STATUS "ccache not found")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(BUILD_TESTS "Build unit tests. Default=OFF" OFF)
option(CPU_BACKEND "Build CPU backend. Default=ON" ON)
option(CUDA_BACKEND "Branch/commit to pull for CUDA backend or `local` if under icicle/backend/cuda. Default=OFF" OFF)

# features that some fields/curves have and some don't.
option(NTT "Build NTT" ON)
option(ECNTT "Build ECNTT" ON)
option(MSM "Build MSM" ON)
option(G2 "Build G2 MSM" ON)
option(EXT_FIELD "Build extension field" ON)
option(HASH "Build hashes and tree builders" ON)
option(POSEIDON "Build poseidon hash" ON)
option(SANITIZE "Enable memory address sanitizer" OFF)

# address sanitizer
if(SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

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
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Specify the installation rules
install(TARGETS icicle_device
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if((DEFINED CURVE) AND (DEFINED FIELD))
  if(NOT ("${CURVE}" STREQUAL "${FIELD}"))
    message(FATAL_ERROR "CURVE and FIELD should not be defined at the same time unless they are equal")
  endif()
endif()

# curve is building the scalar field too
if(CURVE)
  set(CURVE_INDEX -1)
  set(FEATURES_STRING "")
  check_curve(${CURVE} CURVE_INDEX FEATURES_STRING)
  setup_curve_target(${CURVE} ${CURVE_INDEX} ${FEATURES_STRING})
elseif(FIELD)
  set(FIELD_INDEX -1)
  set(FEATURES_STRING "")
  check_field(${FIELD} FIELD_INDEX FEATURES_STRING)
  setup_field_target(${FIELD} ${FIELD_INDEX} ${FEATURES_STRING})
endif()

if (HASH)
  setup_hash_target()
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
    message(STATUS "Fetching cuda backend from ${CUDA_BACKEND_URL}:${CUDA_BACKEND}")
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

