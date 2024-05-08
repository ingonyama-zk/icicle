
include(GoogleTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(runner runner.cu) # TODO link to icicle rather than including source files
target_include_directories(runner PUBLIC ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(runner GTest::gtest_main)
set_target_properties(runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# polynomial test-bench
set(POLY_UNSUPPORTED_FIELD grumpkin)
if (NOT FIELD IN_LIST POLY_UNSUPPORTED_FIELD)
    add_executable(polynomial_tb polynomial_test.cu)
    target_link_libraries(polynomial_tb GTest::gtest_main icicle_field pthread)
    if (CURVE)
        target_link_libraries(polynomial_tb icicle_curve)
    endif()
    set_target_properties(polynomial_tb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    gtest_discover_tests(polynomial_tb)
endif()

enable_testing()

gtest_discover_tests(runner)
