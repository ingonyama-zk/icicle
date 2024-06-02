
message("icicle_v3 tests enabled")

include(GoogleTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Force load backend libraries even though they the symbols are not referenced by icicle or the app
# TODO Yuval: implement dynamic loading and avoid this compiler specific thing
# message("CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
# set(FORCE_LOAD_START "")
# set(FORCE_LOAD_END "")
# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#     set(FORCE_LOAD_START "-Wl,-force_load")
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     set(FORCE_LOAD_START "-Wl,--whole-archive")
#     set(FORCE_LOAD_END "-Wl,--no-whole-archive")
# else()
#     message(FATAL_ERROR "This CMake configuration only supports Clang and GCC.")
# endif()

# device API test
add_executable(test_device_api test_device_api.cpp)
target_include_directories(test_device_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_device_api PUBLIC GTest::gtest_main icicle_device)

#field API test
add_executable(test_field_api test_field_api.cpp)
target_include_directories(test_field_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_field_api PUBLIC GTest::gtest_main icicle_device icicle_field)

#curve API test
add_executable(test_curve_api test_curve_api.cpp)
target_include_directories(test_curve_api PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_curve_api PUBLIC GTest::gtest_main icicle_device icicle_curve)


if(BUILD_CPU_BE)
    target_link_libraries(test_device_api PUBLIC icicle_cpu_device dl)
    target_link_libraries(test_field_api PUBLIC icicle_cpu_device icicle_cpu_field dl)
    target_link_libraries(test_curve_api PUBLIC icicle_cpu_device icicle_cpu_curve dl)
endif()

if(BUILD_CUDA_BE)
    target_link_libraries(test_device_api PUBLIC icicle_cuda_device dl)
    target_link_libraries(test_field_api PUBLIC icicle_cuda_device icicle_cuda_field dl)
    target_link_libraries(test_curve_api PUBLIC icicle_cuda_device icicle_cuda_curve dl)
endif()


enable_testing()
gtest_discover_tests(test_device_api)
gtest_discover_tests(test_field_api)
gtest_discover_tests(test_curve_api)