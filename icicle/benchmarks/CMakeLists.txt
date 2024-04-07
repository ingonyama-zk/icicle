cmake_minimum_required(VERSION 3.18)

include(../cmake/Common.cmake)
include(../cmake/CurvesCommon.cmake)
include(../cmake/FieldsCommon.cmake)

set_env()
set_gpu_env()

project(icicle_benchmarks LANGUAGES CUDA CXX)

if(DEFINED CURVE)
  if(DEFINED FIELD)
    message( FATAL_ERROR "CURVE and FIELD cannot be defined at the same time" )
  endif ()
  check_curve()
elseif(DEFINED FIELD)
  check_field()
endif ()

add_executable(
  benches
  benches.cu
)

target_link_libraries(benches benchmark::benchmark)

find_package(benchmark REQUIRED)
