set(TARGET icicle_hash)

set(SRC ${CMAKE_SOURCE_DIR}/src/hash)

set(HASH_SOURCE ${SRC}/keccak/extern.cu)
list(APPEND HASH_SOURCE ${SRC}/blake2s/extern.cu)

add_library(${TARGET} STATIC ${HASH_SOURCE})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_hash")