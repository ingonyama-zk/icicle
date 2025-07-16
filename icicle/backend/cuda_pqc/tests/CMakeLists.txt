include(GoogleTest)
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

# Disable installation of gtest and gmock
set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Prevent building gtest/gmock by default
foreach(target IN ITEMS gtest gtest_main gmock gmock_main)
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach(target)

enable_testing()

# Common include directories for CUDA tests
set(CUDA_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_SOURCE_DIR}/tests/
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/
    ${CMAKE_CURRENT_SOURCE_DIR}/)

# Define a compile definition for tests to avoid multiple definitions
add_definitions(-DICICLE_CUDA_TEST_ONLY)

# Create a custom target to copy the libraries to a known location
add_custom_target(copy_pqc_libs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        $<TARGET_FILE:icicle_pqc>
        ${CMAKE_BINARY_DIR}/lib/$<TARGET_FILE_NAME:icicle_pqc>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        $<TARGET_FILE:icicle_device>
        ${CMAKE_BINARY_DIR}/lib/$<TARGET_FILE_NAME:icicle_device>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        $<TARGET_FILE:icicle_backend_cuda_pqc>
        ${CMAKE_BINARY_DIR}/lib/$<TARGET_FILE_NAME:icicle_backend_cuda_pqc>
    DEPENDS icicle_pqc icicle_device icicle_backend_cuda_pqc
)

# Helper macro to add a CUDA test
macro(add_cuda_test name)
    add_executable(${name} ml_kem/${name}.cu)
    target_include_directories(${name} PRIVATE ${CUDA_TEST_INCLUDE_DIRS})
    target_compile_definitions(${name} PRIVATE ICICLE_CUDA_TEST_ONLY BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend/cuda_pqc")
    target_link_libraries(${name} PRIVATE gtest_main CUDA::cudart)
    # ensure copy
    add_dependencies(${name} copy_pqc_libs)
    if(ICICLE_STATIC_LINK)
        if(UNIX AND NOT APPLE)
            target_link_options(${name} PRIVATE "LINKER:--whole-archive")
            target_link_libraries(${name} PRIVATE
                ${CMAKE_BINARY_DIR}/lib/libicicle_pqc.a
                ${CMAKE_BINARY_DIR}/lib/libicicle_backend_cuda_pqc.a
                ${CMAKE_BINARY_DIR}/lib/libicicle_device.a)
            target_link_options(${name} PRIVATE "LINKER:--no-whole-archive")
        elseif(APPLE)
            target_link_options(${name} PRIVATE "LINKER:-force_load" ${CMAKE_BINARY_DIR}/lib/libicicle_pqc.a)
            target_link_options(${name} PRIVATE "LINKER:-force_load" ${CMAKE_BINARY_DIR}/lib/libicicle_backend_cuda_pqc.a)
        elseif(MSVC)
            target_link_options(${name} PRIVATE "/WHOLEARCHIVE:${CMAKE_BINARY_DIR}/lib/libicicle_pqc.a")
            target_link_options(${name} PRIVATE "/WHOLEARCHIVE:${CMAKE_BINARY_DIR}/lib/libicicle_backend_cuda_pqc.a")
        else()
            target_link_libraries(${name} PRIVATE ${CMAKE_BINARY_DIR}/lib/libicicle_pqc.a ${CMAKE_BINARY_DIR}/lib/libicicle_backend_cuda_pqc.a)
        endif()
        target_link_libraries(${name} PRIVATE ${CMAKE_BINARY_DIR}/lib/libicicle_device.a dl pthread)
    else()
        target_link_libraries(${name} PRIVATE icicle_backend_cuda_pqc icicle_pqc icicle_device)
    endif()
    gtest_discover_tests(${name})
endmacro()

# Add all CUDA tests
add_cuda_test(test_kyber)
add_cuda_test(test_ml_kem)
add_cuda_test(test_kyber_hashes)
add_cuda_test(test_ml_kem_keygen)
add_cuda_test(test_ml_kem_encaps)
add_cuda_test(test_ml_kem_decaps)
add_cuda_test(test_pke_encrypt)
add_cuda_test(test_pke_decrypt)
