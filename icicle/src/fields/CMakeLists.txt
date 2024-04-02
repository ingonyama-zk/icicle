set(SUPPORTED_FIELDS bn254;bls12_381;bls12_377;bw6_761;grumpkin;babybear;goldilocks;mersenne31)

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

set(TARGET icicle_field_${FIELD})

add_subdirectory(${FIELD})
add_library(${TARGET} STATIC ${FIELD_SOURCE})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/src/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_field_${FIELD}")
target_compile_definitions(${TARGET} PRIVATE FIELD=${FIELD})
