cmake_minimum_required(VERSION 3.21)

# Additive HIP-PQC backend. Mirrors backend/cuda_pqc but registers the
# "HIP-PQC" device and compiles the ML-KEM kernels with the HIP (ROCm)
# toolchain. The NVIDIA "CUDA-PQC" backend is untouched.

project(icicle_hip_pqc_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# enable_language(HIP) honors an explicit -DCMAKE_HIP_ARCHITECTURES, otherwise
# auto-detects the host GPU(s), and errors on a build host with no AMD GPU.
enable_language(HIP)

find_package(hip REQUIRED)

set(HIP_PQC_SOURCES
  src/cuda_pqc_device_api.cu
  src/ml_kem/cuda_pqc_ml_kem.cu)

# Keep the CUDA spelling of the sources and compile them as HIP.
set_source_files_properties(${HIP_PQC_SOURCES} PROPERTIES LANGUAGE HIP)

if(ICICLE_STATIC_LINK)
  add_library(icicle_backend_hip_pqc STATIC ${HIP_PQC_SOURCES})
else()
  add_library(icicle_backend_hip_pqc SHARED ${HIP_PQC_SOURCES})
endif()

target_include_directories(icicle_backend_hip_pqc PRIVATE include)

# Force-include the cuda* -> hip* compatibility shim ahead of every HIP TU so
# the CUDA-spelled runtime calls and warp-synchronous intrinsics resolve to
# their wave-agnostic HIP equivalents.
target_compile_options(icicle_backend_hip_pqc PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-include${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_to_hip.h>)

target_link_libraries(icicle_backend_hip_pqc PRIVATE hip::device)

# The shared HIP-PQC backend is loaded at runtime by the icicle frontend's
# backend loader (icicle_load_backend_from_env_or_default), which dlopens it
# from ICICLE_BACKEND_INSTALL_DIR with RTLD_LOCAL. It is therefore NOT linked
# into icicle_pqc as a NEEDED dependency: doing so would load it a second time
# and the static REGISTER_DEVICE_API initializer would throw on duplicate
# registration. For a static build the interface library whole-archives the
# backend so its self-registration is retained in the final executable.
if(ICICLE_STATIC_LINK)
  add_library(icicle_backend_hip_pqc_interface INTERFACE)
  target_link_libraries(icicle_backend_hip_pqc_interface INTERFACE
    "-Wl,--whole-archive"
    icicle_backend_hip_pqc
    "-Wl,--no-whole-archive")
  list(APPEND PQC_BACKEND_TARGETS_INTERFACE icicle_backend_hip_pqc_interface)
  set(PQC_BACKEND_TARGETS_INTERFACE "${PQC_BACKEND_TARGETS_INTERFACE}" PARENT_SCOPE)
endif()

# Build dependency only (so icicle_pqc triggers the backend build); for the
# shared build the backend is consumed via runtime dlopen, not linkage.
list(APPEND PQC_BACKEND_TARGETS icicle_backend_hip_pqc)
set(PQC_BACKEND_TARGETS "${PQC_BACKEND_TARGETS}" PARENT_SCOPE)

set(_hip_pqc_install_targets icicle_backend_hip_pqc)
if(ICICLE_STATIC_LINK)
  list(APPEND _hip_pqc_install_targets icicle_backend_hip_pqc_interface)
endif()
install(TARGETS ${_hip_pqc_install_targets}
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
