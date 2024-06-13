cmake_minimum_required(VERSION 3.18)

include(cmake/Common.cmake)

# when built as subdir of icicle it is already defined
if (NOT DEFINED NTT)
    option(NTT "Build NTT" ON)
endif()

project(icicle_cuda_backend LANGUAGES CUDA CXX)

set_env()
set_gpu_env()

find_package(CUDAToolkit REQUIRED)

# device API library
add_library(icicle_cuda_device SHARED src/cuda_device_api.cu)
target_include_directories(icicle_cuda_device PRIVATE include)
target_link_libraries(icicle_cuda_device PUBLIC icicle_device)
target_include_directories(icicle_cuda_device PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_device PRIVATE ${CUDA_LIBRARIES}) # Link to CUDA

# field API library
add_library(icicle_cuda_field SHARED 
    src/field/cuda_vec_ops.cu
    src/field/cuda_mont.cu
    )
if(NTT)
    target_sources(icicle_cuda_field PUBLIC 
        src/ntt/cuda_ntt.cu 
        src/ntt/mixed_radix_ntt.cu
    )
endif()
target_include_directories(icicle_cuda_field PRIVATE include)
target_link_libraries(icicle_cuda_field PUBLIC icicle_device icicle_field)
set_target_properties(icicle_cuda_field PROPERTIES OUTPUT_NAME "icicle_cuda_field_${FIELD}")
target_include_directories(icicle_cuda_field PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(icicle_cuda_field PRIVATE ${CUDA_LIBRARIES}) # Link to CUDA

# curve API library
if (CURVE)
    add_library(icicle_cuda_curve SHARED 
        src/curve/cuda_msm.cu
        src/curve/cuda_mont.cu
        src/ntt/cuda_ecntt.cu
    )
    target_include_directories(icicle_cuda_curve PRIVATE include)
    target_link_libraries(icicle_cuda_curve PUBLIC icicle_device icicle_curve)
    set_target_properties(icicle_cuda_curve PROPERTIES OUTPUT_NAME "icicle_cuda_curve_${FIELD}")
    target_include_directories(icicle_cuda_curve PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(icicle_cuda_curve PRIVATE ${CUDA_LIBRARIES}) # Link to CUDA
endif()