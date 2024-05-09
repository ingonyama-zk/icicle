set(TARGET icicle_hash)

add_library(${TARGET} STATIC keccak/keccak.cpp)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_hash")
target_compile_features(${TARGET} PUBLIC cxx_std_17)