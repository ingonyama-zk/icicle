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
    if (POSEIDON)
        target_sources(icicle_field PRIVATE src/hash/cpu_poseidon.cpp)
    endif()
    if (POSEIDON2)
        target_sources(icicle_field PRIVATE src/hash/cpu_poseidon2.cpp)
    endif()
    if(SUMCHECK)
        # Sumcheck relies on hash so this is mandatory
        if(HASH)
        target_sources(icicle_field PRIVATE src/field/cpu_sumcheck.cpp)
        else()
            message(FATAL_ERROR "SUMCHECK is enabled, but HASH is not enabled. Please enable HASH to proceed.")
        endif()
    endif()
    target_include_directories(icicle_field PRIVATE include)
endif() # FIELD

# curve API library
if (CURVE)
    target_sources(icicle_curve PRIVATE
        src/curve/cpu_mont_conversion.cpp
    )
    if (MSM)
        target_sources(icicle_curve PRIVATE src/curve/cpu_msm.cpp)
    endif()
    if (ECNTT)
        target_sources(icicle_curve PRIVATE src/curve/cpu_ecntt.cpp)
    endif()
    # incdir is PUBLIC because config-extension headers are exposed to app
    target_include_directories(icicle_curve PUBLIC include)
endif()

if (HASH)
    target_sources(icicle_hash PRIVATE
        src/hash/cpu_keccak.cpp
        src/hash/cpu_blake2s.cpp
        src/hash/cpu_merkle_tree.cpp
    )
    target_include_directories(icicle_hash PUBLIC include)
endif()


