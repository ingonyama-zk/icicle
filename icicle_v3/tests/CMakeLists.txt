
include(GoogleTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# device API test
add_executable(test_device_api test_device_api.cpp)
target_include_directories(test_device_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_device_api PUBLIC GTest::gtest_main icicle_device)
gtest_discover_tests(test_device_api)

#field API test
add_executable(test_field_api test_field_api.cpp)
target_include_directories(test_field_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_field_api PUBLIC GTest::gtest_main icicle_device icicle_field)
gtest_discover_tests(test_field_api)

#curve API test
if (CURVE)
    add_executable(test_curve_api test_curve_api.cpp)
    target_include_directories(test_curve_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_curve_api PUBLIC GTest::gtest_main icicle_device icicle_field icicle_curve)
    gtest_discover_tests(test_curve_api)
endif()

add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend")

