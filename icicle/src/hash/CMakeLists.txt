set(TARGET icicle_hash)

add_library(${TARGET} STATIC keccak/keccak.cu)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/src/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_hash")