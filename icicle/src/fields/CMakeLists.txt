set(SUPPORTED_FIELDS bn254;bls12_381;bls12_377;bw6_761;grumpkin;babybear;goldilocks;mersenne31)
SET(SUPPORTED_FIELDS_WITHOUT_NTT grumpkin)

set(IS_FIELD_SUPPORTED FALSE)
set(I 0)
foreach (SUPPORTED_FIELD ${SUPPORTED_FIELDS})
  math(EXPR I "${I} + 1")
  if (FIELD STREQUAL SUPPORTED_FIELD)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFIELD_ID=${I}")
    set(IS_FIELD_SUPPORTED TRUE)
  endif ()
endforeach()

if (NOT IS_FIELD_SUPPORTED)
  message( FATAL_ERROR "The value of FIELD variable: ${FIELD} is not one of the supported fields: ${SUPPORTED_FIELD}" )
endif ()

if (EXT_FIELD)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DEXT_FIELD")
endif ()

if (BUILD_BASE_FIELD)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBASE_FIELD")
endif ()

set(TARGET icicle_field_${FIELD})

set(SRC ${CMAKE_SOURCE_DIR}/src)

set(FIELD_SOURCE ${SRC}/fields/field.cu)
list(APPEND FIELD_SOURCE ${SRC}/vec_ops/extern.cu)
list(APPEND FIELD_SOURCE ${SRC}/fields/mont.cu)
list(APPEND FIELD_SOURCE ${SRC}/poseidon/poseidon.cu)
list(APPEND FIELD_SOURCE ${SRC}/poseidon/tree/merkle.cu)

if (NOT FIELD IN_LIST SUPPORTED_FIELDS_WITHOUT_NTT)
    list(APPEND FIELD_SOURCE ${SRC}/ntt/extern.cu)
    list(APPEND FIELD_SOURCE ${SRC}/ntt/kernel_ntt.cu)
endif()

add_library(${TARGET} STATIC ${FIELD_SOURCE})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_field_${FIELD}")
target_compile_definitions(${TARGET} PRIVATE FIELD=${FIELD})
