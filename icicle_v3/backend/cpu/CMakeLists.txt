cmake_minimum_required(VERSION 3.18)

project(icicle_cpu_backend)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO Yuval: handle incdir when backend is separated from frontend
# TODO Yuval: dll?

include_directories(${CMAKE_SOURCE_DIR}/include)

# device API library
add_library(icicle_cpu_device SHARED src/cpu_device_api.cpp)
target_link_libraries(icicle_cpu_device PUBLIC icicle_device)

# field API library
add_library(icicle_cpu_field SHARED 
    src/field/cpu_vec_ops.cpp 
    src/field/cpu_ntt.cpp
    src/field/cpu_matrix_ops.cpp
)
target_link_libraries(icicle_cpu_field PUBLIC icicle_device icicle_field)
set_target_properties(icicle_cpu_field PROPERTIES OUTPUT_NAME "icicle_cpu_field_${FIELD}")

# curve API library
add_library(icicle_cpu_curve SHARED src/curve/cpu_msm.cpp)
target_link_libraries(icicle_cpu_curve PUBLIC icicle_device icicle_curve)
set_target_properties(icicle_cpu_curve PROPERTIES OUTPUT_NAME "icicle_cpu_curve_${CURVE}")
