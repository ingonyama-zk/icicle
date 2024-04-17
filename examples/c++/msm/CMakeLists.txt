cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
endif ()
project(example LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

add_executable(
  example
  example.cu
)
target_include_directories(example PRIVATE "../../../icicle/include")
target_link_libraries(example ${CMAKE_SOURCE_DIR}/../../../icicle/build/lib/libingo_curve_bn254.a)
target_link_libraries(example ${CMAKE_SOURCE_DIR}/../../../icicle/build/lib/libingo_field_bn254.a)
set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
