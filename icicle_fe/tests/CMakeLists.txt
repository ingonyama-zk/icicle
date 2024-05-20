
message("icicle_fe tests enabled")

include(GoogleTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_device_api test_device_api.cpp)
target_include_directories(test_device_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_device_api GTest::gtest_main icicle_device icicle_cpu_backend)

enable_testing()
gtest_discover_tests(test_device_api)