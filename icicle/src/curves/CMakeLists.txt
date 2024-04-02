set(SUPPORTED_CURVES bn254;bls12_381;bls12_377;bw6_761;grumpkin)

set(IS_CURVE_SUPPORTED FALSE)
set(I 0)
foreach (SUPPORTED_CURVE ${SUPPORTED_CURVES})
  math(EXPR I "${I} + 1")
  if (CURVE STREQUAL SUPPORTED_CURVE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCURVE_ID=${I}")
    set(IS_CURVE_SUPPORTED TRUE)
  endif ()
endforeach()

if (NOT IS_CURVE_SUPPORTED)
  message( FATAL_ERROR "The value of CURVE variable: ${CURVE} is not one of the supported fields: ${SUPPORTED_CURVES}" )
endif ()

if (G2)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DG2 -DEXT_FIELD")
endif ()

set(TARGET icicle_curve_${CURVE})
set(FIELD_TARGET icicle_field_${FIELD})

add_subdirectory(${CURVE})
add_library(${TARGET} STATIC ${CURVE_SOURCE})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include/)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "ingo_curve_${CURVE}")
target_compile_definitions(${TARGET} PRIVATE CURVE=${CURVE})
target_link_libraries(${TARGET} PRIVATE ${FIELD_TARGET})