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
endforeach()

enable_testing()

if(PQC)
    # Common include directories for CUDA tests
    set(CUDA_TEST_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/ 
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
        ${CMAKE_CURRENT_SOURCE_DIR}/ml_kem/
        ${CMAKE_SOURCE_DIR}/tests/
    )

    # Helper macro to define test targets
    function(add_cuda_test name)
        add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/ml_kem/${name}.cu)
        target_include_directories(${name} PRIVATE ${CUDA_TEST_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE GTest::gtest_main icicle_pqc)
        gtest_discover_tests(${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ml_kem)
    endfunction()

    # Add all CUDA tests
    add_cuda_test(test_kyber)
    add_cuda_test(test_ml_kem_keygen)
    add_cuda_test(test_pke_encrypt)
    add_cuda_test(test_pke_decrypt)
    add_cuda_test(test_kyber_hashes)
    add_cuda_test(test_ml_kem)
    add_cuda_test(test_ml_kem_encaps)
    add_cuda_test(test_ml_kem_decaps)
endif()
