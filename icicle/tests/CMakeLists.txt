
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

# Helper macro to add system libs for static linking
macro(icicle_add_system_libs target)
  if(ICICLE_STATIC_LINK)
    target_link_libraries(${target} PRIVATE dl pthread)
  endif()
endmacro()

# device API test
add_executable(test_device_api test_device_api.cpp)
target_include_directories(test_device_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
target_link_libraries(test_device_api PRIVATE GTest::gtest_main)
icicle_add_system_libs(test_device_api)
gtest_discover_tests(test_device_api)

# Added backend libs when static linking
if(ICICLE_STATIC_LINK)
  # system libs
  target_link_libraries(test_device_api PRIVATE dl pthread)
  if(UNIX AND NOT APPLE)
    target_link_libraries(test_device_api PRIVATE "-Wl,--whole-archive" icicle_backend_cuda_pqc icicle_pqc ${CMAKE_BINARY_DIR}/lib/libicicle_device.a "-Wl,--no-whole-archive")
  elseif(APPLE)
    target_link_options(test_device_api PRIVATE "LINKER:-force_load" $<TARGET_FILE:icicle_backend_cuda_pqc>)
  elseif(MSVC)
    target_link_options(test_device_api PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:icicle_backend_cuda_pqc>")
  endif()
endif()

#field API test
if (FIELD)
    add_executable(test_field_api test_field_api.cpp)
    target_include_directories(test_field_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_field_api PRIVATE GTest::gtest_main icicle_device icicle_field)
    icicle_add_system_libs(test_field_api)
    gtest_discover_tests(test_field_api)

    if (NTT)
        add_executable(test_polynomial_api test_polynomial_api.cpp)
        target_include_directories(test_polynomial_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
        target_link_libraries(test_polynomial_api PRIVATE GTest::gtest_main icicle_field)
        icicle_add_system_libs(test_polynomial_api)
        gtest_discover_tests(test_polynomial_api)    
    endif()

    if(SUMCHECK OR HASH OR FRI)
        target_link_libraries(test_field_api PRIVATE icicle_hash)
        if(ICICLE_STATIC_LINK)
          target_link_libraries(test_field_api PRIVATE icicle_device)
        endif()
    endif()
endif()

#curve API test
if (CURVE)
    add_executable(test_curve_api test_curve_api.cpp)
    target_include_directories(test_curve_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_curve_api PRIVATE GTest::gtest_main icicle_device icicle_field icicle_curve)
    icicle_add_system_libs(test_curve_api)
    gtest_discover_tests(test_curve_api)

    if (NTT)        
        target_link_libraries(test_polynomial_api PRIVATE GTest::gtest_main icicle_curve)
        if(ICICLE_STATIC_LINK)
          target_link_libraries(test_polynomial_api PRIVATE icicle_device icicle_field)
        endif()
    endif()
endif()

#hash API test
if (HASH)
    add_executable(test_hash_api test_hash_api.cpp)
    target_include_directories(test_hash_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_hash_api PRIVATE GTest::gtest_main icicle_hash)
    icicle_add_system_libs(test_hash_api)
    gtest_discover_tests(test_hash_api)
    if (POSEIDON AND (FIELD OR CURVE))
        target_link_libraries(test_hash_api PRIVATE icicle_field)
    endif()
    if (POSEIDON2 AND (FIELD OR CURVE))
        target_link_libraries(test_hash_api PRIVATE icicle_field)
    endif()
    # Added backend libs when static linking
    if(ICICLE_STATIC_LINK)
        target_link_libraries(test_hash_api PRIVATE dl pthread)
        if(UNIX AND NOT APPLE)
            target_link_libraries(test_hash_api PRIVATE "-Wl,--whole-archive" icicle_backend_cuda_pqc icicle_pqc ${CMAKE_BINARY_DIR}/lib/libicicle_device.a "-Wl,--no-whole-archive")
        elseif(APPLE)
            target_link_options(test_hash_api PRIVATE "LINKER:-force_load" $<TARGET_FILE:icicle_backend_cuda_pqc>)
        elseif(MSVC)
            target_link_options(test_hash_api PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:icicle_backend_cuda_pqc>")
        endif()
    endif()
endif()

if (RING)
    add_executable(test_ring_api test_ring_api.cpp)
    target_include_directories(test_ring_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    target_link_libraries(test_ring_api PRIVATE GTest::gtest_main icicle_device icicle_ring)
    icicle_add_system_libs(test_ring_api)
    gtest_discover_tests(test_ring_api)
endif()

if (PQC)
    add_executable(test_pqc_api test_pqc_api.cpp)
    target_include_directories(test_pqc_api PRIVATE ${CMAKE_SOURCE_DIR}/include/)
    
    if (ICICLE_STATIC_LINK)
        target_link_libraries(test_pqc_api PRIVATE GTest::gtest_main)
        # system libs
        target_link_libraries(test_pqc_api PRIVATE dl pthread)
        # Pull in all objects from both icicle_pqc and backend libs to keep registration
        if(UNIX AND NOT APPLE)
            target_link_libraries(test_pqc_api PRIVATE "-Wl,--whole-archive" icicle_pqc icicle_backend_cuda_pqc ${CMAKE_BINARY_DIR}/lib/libicicle_device.a "-Wl,--no-whole-archive")
        elseif(APPLE)
            target_link_options(test_pqc_api PRIVATE "LINKER:-force_load" $<TARGET_FILE:icicle_pqc>)
            target_link_options(test_pqc_api PRIVATE "LINKER:-force_load" $<TARGET_FILE:icicle_backend_cuda_pqc>)
        elseif(MSVC)
            target_link_options(test_pqc_api PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:icicle_pqc>")
            target_link_options(test_pqc_api PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:icicle_backend_cuda_pqc>")
        endif()
    else()
        target_link_libraries(test_pqc_api PRIVATE GTest::gtest_main icicle_device icicle_pqc icicle_backend_cuda_pqc)
    endif()
    
    icicle_add_system_libs(test_pqc_api)
    gtest_discover_tests(test_pqc_api)
endif()