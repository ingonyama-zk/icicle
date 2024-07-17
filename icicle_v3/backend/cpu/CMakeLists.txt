cmake_minimum_required(VERSION 3.18)

# CPU backend is built directly into icicle library

target_sources(icicle_device PRIVATE src/cpu_device_api.cpp)

# field API library
if (FIELD)
    target_sources(icicle_field PRIVATE
        src/field/cpu_vec_ops.cpp        
    )
    if (NTT)
        target_sources(icicle_field PRIVATE src/field/cpu_ntt.cpp src/polynomials/cpu_polynomial_backend.cpp)
    endif()    
    target_include_directories(icicle_field PRIVATE include)
endif() # FIELD

# curve API library
if (CURVE)
target_sources(icicle_curve PRIVATE
        src/curve/cpu_msm.cpp
        src/curve/cpu_mont_conversion.cpp
    )
    if (ECNTT)
        target_sources(icicle_curve PRIVATE src/curve/cpu_ecntt.cpp)
    endif()
    # incdir is PUBLIC because config-extension headers are exposed to app
    target_include_directories(icicle_curve PUBLIC include)
endif()


