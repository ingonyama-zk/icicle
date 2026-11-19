include(GoogleTest)
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

foreach(target IN ITEMS gtest gtest_main gmock gmock_main)
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach(target)

enable_testing()

set(HIP_PQC_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_SOURCE_DIR}/tests/
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/
    ${CMAKE_CURRENT_SOURCE_DIR}/)

add_definitions(-DICICLE_CUDA_TEST_ONLY)

# Copy the libraries needed to load the backend at runtime.
add_custom_target(copy_hip_pqc_libs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:icicle_pqc>
        ${CMAKE_BINARY_DIR}/lib/$<TARGET_FILE_NAME:icicle_pqc>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:icicle_device>
        ${CMAKE_BINARY_DIR}/lib/$<TARGET_FILE_NAME:icicle_device>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:icicle_backend_hip_pqc>
        ${CMAKE_BINARY_DIR}/lib/$<TARGET_FILE_NAME:icicle_backend_hip_pqc>
    DEPENDS icicle_pqc icicle_device icicle_backend_hip_pqc
)

# Reuse the cuda_pqc backend's NIST KAT vectors, copied into the build dir at
# configure time, instead of shipping a separate copy under this backend.
add_custom_target(copy_hip_test_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_pqc/tests/ml_kem/test_data
        ${CMAKE_CURRENT_BINARY_DIR}/test_data
    COMMENT "Copying test data to build directory"
)

# Targets carry a hip_ prefix so they do not collide with the cuda_pqc tests
# when both backends are configured together.
macro(add_hip_test test)
    set(name hip_${test})
    add_executable(${name} ml_kem/${test}.cu)
    set_source_files_properties(ml_kem/${test}.cu PROPERTIES LANGUAGE HIP)
    target_include_directories(${name} PRIVATE ${HIP_PQC_TEST_INCLUDE_DIRS})
    target_compile_definitions(${name} PRIVATE ICICLE_CUDA_TEST_ONLY BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend/hip_pqc")
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-include${CMAKE_CURRENT_SOURCE_DIR}/../include/cuda_to_hip.h>)
    target_link_libraries(${name} PRIVATE gtest_main hip::device)
    add_dependencies(${name} copy_hip_pqc_libs copy_hip_test_data)
    if(ICICLE_STATIC_LINK)
        target_link_options(${name} PRIVATE "LINKER:--whole-archive")
        target_link_libraries(${name} PRIVATE
            ${CMAKE_BINARY_DIR}/lib/libicicle_pqc.a
            ${CMAKE_BINARY_DIR}/lib/libicicle_backend_hip_pqc.a
            ${CMAKE_BINARY_DIR}/lib/libicicle_device.a)
        target_link_options(${name} PRIVATE "LINKER:--no-whole-archive")
        target_link_libraries(${name} PRIVATE ${CMAKE_BINARY_DIR}/lib/libicicle_device.a dl pthread)
    else()
        # The backend is loaded at runtime from BACKEND_BUILD_DIR; linking it
        # here as well would register the "HIP-PQC" device twice (the loader
        # dlopens the same .so with RTLD_LOCAL, re-running its static init).
        target_link_libraries(${name} PRIVATE icicle_pqc icicle_device)
    endif()
    gtest_discover_tests(${name})
endmacro()

add_hip_test(test_kyber)
add_hip_test(test_ml_kem)
add_hip_test(test_kyber_hashes)
add_hip_test(test_ml_kem_keygen)
add_hip_test(test_ml_kem_encaps)
add_hip_test(test_ml_kem_decaps)
add_hip_test(test_pke_encrypt)
add_hip_test(test_pke_decrypt)
