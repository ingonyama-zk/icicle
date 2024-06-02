cmake_minimum_required(VERSION 3.18)

project(icicle_cuda_backend LANGUAGES CUDA CXX)

include(cmake/Common.cmake)

set_env()
set_gpu_env()

find_package(CUDA REQUIRED)

# device API library
add_library(icicle_cuda_device SHARED src/cuda_device_api.cu)
target_include_directories(icicle_cuda_device PRIVATE include)
target_link_libraries(icicle_cuda_device PUBLIC icicle_device)
target_include_directories(icicle_cuda_device PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_device PRIVATE ${CUDA_LIBRARIES}) # Link to CUDA

# field API library
add_library(icicle_cuda_field SHARED src/field/cuda_vec_ops.cu)
target_include_directories(icicle_cuda_field PRIVATE include)
target_link_libraries(icicle_cuda_field PUBLIC icicle_device icicle_field)
set_target_properties(icicle_cuda_field PROPERTIES OUTPUT_NAME "icicle_cuda_field_${FIELD}")
target_include_directories(icicle_cuda_field PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_field PRIVATE ${CUDA_LIBRARIES}) # Link to CUDA

# curve API library
add_library(icicle_cuda_curve SHARED src/curve/cuda_msm.cu)
target_include_directories(icicle_cuda_curve PRIVATE include)
target_link_libraries(icicle_cuda_curve PUBLIC icicle_device icicle_curve)
set_target_properties(icicle_cuda_curve PROPERTIES OUTPUT_NAME "icicle_cuda_curve_${FIELD}")
target_include_directories(icicle_cuda_curve PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_curve PRIVATE ${CUDA_LIBRARIES}) # Link to CUDA