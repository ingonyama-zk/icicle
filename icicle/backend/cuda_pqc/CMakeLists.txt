cmake_minimum_required(VERSION 3.18)

# TODO: share cmake code with the closed CUDA backend ? worth it?
include(cmake/Common.cmake)
find_cuda_compiler()

project(icicle_cuda_pqc_backend LANGUAGES CUDA CXX)

set_env()
set_gpu_env()

find_package(CUDAToolkit REQUIRED)

# Build CUDA backend as a static lib and wrap it with whole-archive in an INTERFACE target
# Note: whole-archive is used to ensure all symbols are included in the final binary, otherwise registration is skipped
add_library(icicle_backend_cuda_pqc_device STATIC 
src/cuda_pqc_device_api.cu
src/cuda_pqc_ml_kem.cu
)
target_include_directories(icicle_backend_cuda_pqc_device PRIVATE include)
add_library(icicle_backend_cuda_pqc_device_whole INTERFACE)
target_link_libraries(icicle_backend_cuda_pqc_device_whole INTERFACE
"-Wl,--whole-archive" icicle_backend_cuda_pqc_device "-Wl,--no-whole-archive")

# Link into icicle-pqc
target_link_libraries(icicle_pqc PRIVATE icicle_backend_cuda_pqc_device_whole)
target_link_libraries(icicle_pqc PRIVATE cudart)