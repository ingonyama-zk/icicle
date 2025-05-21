cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(example)

function(force_load_static_library target lib)
  if(APPLE)
    get_target_property(lib_path ${lib} LOCATION)
    target_link_libraries(${target} PRIVATE "-Wl,-force_load,${lib_path}")
  else()
    # On Linux use --whole-archive
    target_link_libraries(${target} PRIVATE "-Wl,--whole-archive" ${lib} "-Wl,--no-whole-archive")
  endif()
endfunction()

add_executable(example example.cpp)
target_include_directories(example PRIVATE "../../../icicle/include" "..")
target_link_directories(example PRIVATE "${CMAKE_SOURCE_DIR}/build/icicle")
message("${CMAKE_BINARY_DIR}/icicle")

#### For ios we need to link against the static libraries and force load all symbols to force registration
#### Note: linking backend libs should be quite similar.
#### Note: For rust this logic is in build.rs script
add_library(icicle_curve_bn254 STATIC IMPORTED)
set_target_properties(icicle_curve_bn254 PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/build/icicle/libicicle_curve_bn254.a"
)

add_library(icicle_field_bn254 STATIC IMPORTED)
set_target_properties(icicle_field_bn254 PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/build/icicle/libicicle_field_bn254.a"
)

add_library(icicle_device STATIC IMPORTED)
set_target_properties(icicle_device PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/build/icicle/libicicle_device.a")

add_library(icicle_hash STATIC IMPORTED)
set_target_properties(icicle_hash PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/build/icicle/libicicle_hash.a")

force_load_static_library(example icicle_device)
force_load_static_library(example icicle_hash)
force_load_static_library(example icicle_field_bn254)
force_load_static_library(example icicle_curve_bn254)
###

if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()

