cmake_minimum_required(VERSION 3.18)

project(icicle_cuda_backend LANGUAGES CUDA CXX)

include(cmake/Common.cmake)

set_env()
set_gpu_env()

# TODO Yuval: handle incdir when backend is seperated from frontend
# TODO Yuval: dll?
add_library(icicle_cuda_device STATIC cuda_device_api.cu)
target_link_libraries(icicle_cuda_device PUBLIC icicle_device)

# Link to CUDA
find_package(CUDA REQUIRED)
target_include_directories(icicle_cuda_device PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_device PRIVATE ${CUDA_LIBRARIES})
# Specify the CUDA standard
# set_target_properties(icicle_cuda_device PROPERTIES CUDA_STANDARD 14)