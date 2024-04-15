if (EXT_FIELD)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DEXT_FIELD")
endif ()

SET(SUPPORTED_FIELDS_WITHOUT_NTT grumpkin)

set(TARGET icicle_field)

set(SRC ${CMAKE_SOURCE_DIR}/src)

set(FIELD_SOURCE ${SRC}/fields/extern.cu)
list(APPEND FIELD_SOURCE ${SRC}/vec_ops/extern.cu)
if(EXT_FIELD)
  list(APPEND FIELD_SOURCE ${SRC}/fields/extern_extension.cu)
  list(APPEND FIELD_SOURCE ${SRC}/ntt/extern_extension.cu)
  list(APPEND FIELD_SOURCE ${SRC}/vec_ops/extern_extension.cu)
endif()

set(POLYNOMIAL_SOURCE_FILES 
    ${SRC}/polynomials/polynomials.cu
    ${SRC}/polynomials/cuda_backend/polynomial_cuda_backend.cu)

list(APPEND FIELD_SOURCE ${POLYNOMIAL_SOURCE_FILES})

# TODO: impl poseidon for small fields. note that it needs to be defined for the extension field!
if (DEFINED CURVE)
  list(APPEND FIELD_SOURCE ${SRC}/poseidon/poseidon.cu)
  list(APPEND FIELD_SOURCE ${SRC}/poseidon/tree/merkle.cu)
endif()

if (NOT FIELD IN_LIST SUPPORTED_FIELDS_WITHOUT_NTT)
  list(APPEND FIELD_SOURCE ${SRC}/ntt/extern.cu)
  list(APPEND FIELD_SOURCE ${SRC}/ntt/kernel_ntt.cu)
endif()

add_library(${TARGET} STATIC ${FIELD_SOURCE})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_field_${FIELD}")
target_compile_definitions(${TARGET} PUBLIC FIELD=${FIELD})
