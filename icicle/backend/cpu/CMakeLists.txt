cmake_minimum_required(VERSION 3.18)

# CPU backend is built directly into icicle library

target_sources(icicle_device PRIVATE src/cpu_device_api.cpp)

# Attempt to find OpenMP first
find_package(OpenMP)

if(OpenMP_CXX_FOUND AND NOT DEFINED FORCE_BUILD_OPENMP)
    message(STATUS "Using system OpenMP: ${OpenMP_CXX_INCLUDE_DIRS}, ${OpenMP_CXX_LIBRARIES}")

    # Link to system OpenMP
    target_include_directories(icicle_device PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(icicle_device PUBLIC OpenMP::OpenMP_CXX)

else()
    # Fallback: Build OpenMP using the script
    set(OPENMP_INSTALL_DIR "${CMAKE_SOURCE_DIR}/deps/openmp") # Install OpenMP in deps/openmp
    set(OPENMP_BUILD_SCRIPT "${CMAKE_SOURCE_DIR}/../scripts/build_openmp.sh")
    
    message(STATUS "OpenMP not found or FORCE_BUILD_OPENMP is enabled.")
    message(STATUS "Falling back to building OpenMP from source.")
    message(STATUS "OpenMP will be built and installed in: ${OPENMP_INSTALL_DIR}")

    # Custom command to build OpenMP
    add_custom_command(
        OUTPUT ${OPENMP_INSTALL_DIR}/lib/libomp.a ${OPENMP_INSTALL_DIR}/include/omp.h
        COMMAND bash ${OPENMP_BUILD_SCRIPT} ${OPENMP_INSTALL_DIR}
        COMMENT "Building and installing OpenMP to ${OPENMP_INSTALL_DIR}..."
        VERBATIM
    )

    # Custom target for OpenMP
    add_custom_target(openmp_build
        DEPENDS ${OPENMP_INSTALL_DIR}/lib/libomp.a ${OPENMP_INSTALL_DIR}/include/omp.h
        COMMENT "OpenMP build completed."
    )

    # Link to built OpenMP headers and libraries
    target_include_directories(icicle_device PUBLIC ${OPENMP_INSTALL_DIR}/include)
    target_link_directories(icicle_device PUBLIC ${OPENMP_INSTALL_DIR}/lib)
    target_link_libraries(icicle_device PUBLIC libomp.a)

    # Ensure icicle_device depends on OpenMP build
    add_dependencies(icicle_device openmp_build)
endif()


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

