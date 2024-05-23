cmake_minimum_required(VERSION 3.18)

project(icicle_cuda_backend LANGUAGES CUDA CXX)

include(cmake/Common.cmake)

set_env()
set_gpu_env()

find_package(CUDA REQUIRED)

include_directories(include)

# TODO Yuval: handle incdir when backend is seperated from frontend
# TODO Yuval: dll?
add_library(icicle_cuda_device STATIC src/cuda_device_api.cu)
target_link_libraries(icicle_cuda_device PUBLIC icicle_device)
# Link to CUDA
target_include_directories(icicle_cuda_device PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_device PRIVATE ${CUDA_LIBRARIES})

# field API library
add_library(icicle_cuda_field STATIC src/field/cuda_vec_ops.cu)
target_link_libraries(icicle_cuda_field PUBLIC icicle_device icicle_field)
# Link to CUDA
target_include_directories(icicle_cuda_field PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_field PRIVATE ${CUDA_LIBRARIES})
