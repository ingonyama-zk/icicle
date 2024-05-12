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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -DCURVE_ID=BN254")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

add_executable(
  example
  example.cu
)

set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(example PRIVATE "../../../icicle/include")

# can link to another curve/field by changing the following lib and FIELD_ID
target_link_libraries(example ${CMAKE_SOURCE_DIR}/build/icicle/lib/libingo_field_bn254.a)
target_compile_definitions(example PUBLIC FIELD_ID BN254)



# cmake_minimum_required(VERSION 3.18)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
#     set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
# else()
#     set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
# endif ()
# project(zk-benchmarks LANGUAGES CUDA CXX)

# #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# #set(CMAKE_CUDA_FLAGS_RELEASE "")
# #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# # assume repos icicle ans zk-benchmarks are in the same directory
# set(ICICLE_BASE_PATH "..")
# include_directories("${ICICLE_BASE_PATH}/icicle/icicle/include")
# include_directories("/opt/benchmark/include")

# add_executable(
#   benchmark
#   benchmark.cu
# )

# find_library(CURVE_LIBRARY ingo_curve_bn254 PATHS ${ICICLE_BASE_PATH}/icicle/icicle/build/lib)
# find_library(FIELD_LIBRARY ingo_field_bn254 PATHS ${ICICLE_BASE_PATH}/icicle/icicle/build/lib)
# # find_library(BENCHMARK_LIBRARY benchmark PATHS /usr/local/lib)
# find_library(NVML_LIBRARY nvidia-ml PATHS /usr/local/cuda/targets/x86_64-linux/lib/stubs/ )
# target_link_libraries(benchmark  ${NVML_LIBRARY} ${CURVE_LIBRARY} ${FIELD_LIBRARY})


# set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# cmake_minimum_required(VERSION 3.18)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
#     set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
# else()
#     set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
# endif ()
# project(zk-benchmarks LANGUAGES CUDA CXX)

# #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# #set(CMAKE_CUDA_FLAGS_RELEASE "")
# #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# include_directories("/opt/icicle/icicle")
# include_directories("/opt/benchmark/include")

# add_executable(
#   benchmark
#   benchmark.cu
# )

# find_library(BENCHMARK_LIBRARY benchmark PATHS /usr/local/lib)
# find_library(NVML_LIBRARY nvidia-ml PATHS /usr/local/cuda/targets/x86_64-linux/lib/stubs/ )
# target_link_libraries(benchmark ${BENCHMARK_LIBRARY} ${NVML_LIBRARY})
# set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

