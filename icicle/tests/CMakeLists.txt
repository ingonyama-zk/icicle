
include(GoogleTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

# Disable installation of gtest and gmock
set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Ensure gtest and gmock are not built by default
set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(gmock PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(gmock_main PROPERTIES EXCLUDE_FROM_ALL TRUE)

enable_testing()

# device API test
add_executable(test_device_api test_device_api.cpp)
target_include_directories(test_device_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_device_api PRIVATE GTest::gtest_main icicle_device)
gtest_discover_tests(test_device_api)

#field API test
if (FIELD)
    add_executable(test_field_api test_field_api.cpp)
    target_include_directories(test_field_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_field_api PRIVATE GTest::gtest_main icicle_device icicle_field)
    gtest_discover_tests(test_field_api)

    if (NTT)
        add_executable(test_polynomial_api test_polynomial_api.cpp)
        target_include_directories(test_polynomial_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
        target_link_libraries(test_polynomial_api PRIVATE GTest::gtest_main icicle_field)
        gtest_discover_tests(test_polynomial_api)    
    endif()
endif()

#curve API test
if (CURVE)
    add_executable(test_curve_api test_curve_api.cpp)
    target_include_directories(test_curve_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_curve_api PRIVATE GTest::gtest_main icicle_device icicle_field icicle_curve)
    gtest_discover_tests(test_curve_api)

    if (NTT)        
        target_link_libraries(test_polynomial_api PRIVATE GTest::gtest_main icicle_curve)
    endif()
endif()

#hash API test
if (HASH)
    add_executable(test_hash_api test_hash_api.cpp)
    target_include_directories(test_hash_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_hash_api PRIVATE GTest::gtest_main icicle_device icicle_hash)
    gtest_discover_tests(test_hash_api)
    if (POSEIDON AND (FIELD OR CURVE))
        target_link_libraries(test_hash_api PRIVATE icicle_field)
    endif()
    if (POSEIDON2 AND (FIELD OR CURVE))
        target_link_libraries(test_hash_api PRIVATE icicle_field)
    endif()
endif()
