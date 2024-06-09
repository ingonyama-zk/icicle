cmake_minimum_required(VERSION 3.18)

project(icicle_cpu_backend)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# when built as subdir of icicle it is already defined
if (NOT DEFINED NTT)
    option(NTT "Build NTT" ON)
endif()

# device API library
add_library(icicle_cpu_device SHARED src/cpu_device_api.cpp)
target_link_libraries(icicle_cpu_device PUBLIC icicle_device)

# field API library
add_library(icicle_cpu_field SHARED 
    src/field/cpu_vec_ops.cpp     
    src/field/cpu_matrix_ops.cpp
)
if (NTT)    
    target_sources(icicle_cpu_field PUBLIC src/field/cpu_ntt.cpp)
endif()
target_include_directories(icicle_cpu_field PUBLIC include)
target_link_libraries(icicle_cpu_field PUBLIC icicle_device icicle_field)
set_target_properties(icicle_cpu_field PROPERTIES OUTPUT_NAME "icicle_cpu_field_${FIELD}")

# curve API library
if (CURVE)
    add_library(icicle_cpu_curve SHARED 
        src/curve/cpu_msm.cpp
        src/curve/cpu_mont_conversion.cpp        
    )
    if (ECNTT)
        target_sources(icicle_cpu_curve PRIVATE src/curve/cpu_ecntt.cpp)
    endif()
    target_include_directories(icicle_cpu_curve PUBLIC include)
    target_link_libraries(icicle_cpu_curve PUBLIC icicle_device icicle_curve)
    set_target_properties(icicle_cpu_curve PROPERTIES OUTPUT_NAME "icicle_cpu_curve_${CURVE}")
endif()
