if (EXT_FIELD)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DEXT_FIELD")
endif ()

SET(SUPPORTED_FIELDS_WITHOUT_NTT grumpkin)
SET(SUPPORTED_FIELDS_WITHOUT_POSEIDON2 bls12_381;bls12_377;grumpkin;bw6_761;stark252)

set(TARGET icicle_field)

set(SRC ${CMAKE_SOURCE_DIR}/src)

set(FIELD_SOURCE ${SRC}/fields/extern.cu)
list(APPEND FIELD_SOURCE ${SRC}/vec_ops/extern.cu)
list(APPEND FIELD_SOURCE ${SRC}/merkle-tree/extern.cu)

if(EXT_FIELD)
  list(APPEND FIELD_SOURCE ${SRC}/fields/extern_extension.cu)
  list(APPEND FIELD_SOURCE ${SRC}/ntt/extern_extension.cu)
  list(APPEND FIELD_SOURCE ${SRC}/vec_ops/extern_extension.cu)
endif()

set(POLYNOMIAL_SOURCE_FILES 
    ${SRC}/polynomials/polynomials.cu
    ${SRC}/polynomials/cuda_backend/polynomial_cuda_backend.cu
    ${SRC}/polynomials/polynomials_c_api.cu)

# TODO: impl poseidon for small fields. note that it needs to be defined over the extension field!
if (DEFINED CURVE)
  list(APPEND FIELD_SOURCE ${SRC}/poseidon/extern.cu)
endif()

if (NOT FIELD IN_LIST SUPPORTED_FIELDS_WITHOUT_POSEIDON2)
  list(APPEND FIELD_SOURCE ${SRC}/poseidon2/extern.cu)
endif()

if (NOT FIELD IN_LIST SUPPORTED_FIELDS_WITHOUT_NTT)
  list(APPEND FIELD_SOURCE ${SRC}/ntt/extern.cu)
  list(APPEND FIELD_SOURCE ${SRC}/ntt/kernel_ntt.cu)
  list(APPEND FIELD_SOURCE ${POLYNOMIAL_SOURCE_FILES}) # requires NTT  
endif()

add_library(${TARGET} STATIC ${FIELD_SOURCE})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_field_${FIELD}")
target_compile_definitions(${TARGET} PUBLIC FIELD=${FIELD})
