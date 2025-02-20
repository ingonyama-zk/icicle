cmake_minimum_required(VERSION 3.18)

message(STATUS "Fetching Taskflow v3.8.0 (CPU backend)")


include(FetchContent)
# Temporarily redefine message() to suppress output
macro(message)
endmacro()
FetchContent_Declare(
  Taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG v3.8.0
  GIT_SHALLOW TRUE
)
# Disable unnecessary components
set(TF_BUILD_BENCHMARKS OFF CACHE BOOL "Disable Taskflow benchmarks" FORCE)
set(TF_BUILD_PROFILER OFF CACHE BOOL "Disable Taskflow profiler" FORCE)
set(TF_BUILD_CUDA OFF CACHE BOOL "Disable Taskflow CUDA support" FORCE)
set(TF_BUILD_SYCL OFF CACHE BOOL "Disable Taskflow SYCL support" FORCE)
set(TF_BUILD_TESTS OFF CACHE BOOL "Disable Taskflow tests" FORCE)
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "Disable Taskflow examples" FORCE)

FetchContent_MakeAvailable(Taskflow)
# Use icicle_device as interface for TaskFlow headers
target_include_directories(icicle_device INTERFACE ${Taskflow_SOURCE_DIR})

# Restore the original message behavior
unset(message)

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
        target_sources(icicle_field PRIVATE src/field/cpu_sumcheck.cpp)
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
        src/hash/cpu_blake3.cpp
        src/hash/blake3.c
        src/hash/blake3_dispatch.c
        src/hash/blake3_portable.c
        src/hash/cpu_merkle_tree.cpp
        src/hash/cpu_pow.cpp
    )
    target_include_directories(icicle_hash PUBLIC include)
endif()

if (RING)
    target_sources(icicle_ring PRIVATE src/field/cpu_vec_ops.cpp)
    if (NTT)
        target_sources(icicle_ring PRIVATE src/field/cpu_ntt.cpp)
    endif()
    target_include_directories(icicle_ring PRIVATE include)
endif()
